import torch
import torchmetrics
import wandb
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AdamW
from constants import LABEL_COLUMNS, LEARNING_RATE, WANDB
import torch.nn.functional as F


class Model_BERT_GRU(pl.LightningModule):

    def __init__(self, n_classes: int, bert=None, class_weights=None):
        super().__init__()
        self.bert = bert

        self.hidden_size = 512

        # Additional layers
        self.gru = nn.GRU(self.bert.config.hidden_size, self.hidden_size,
                          bidirectional=True, batch_first=True, num_layers=3)

        # Dropout
        self.dropout = nn.Dropout(0.2)

        # Fully connected
        self.fc = nn.Linear(self.hidden_size * 2, n_classes)

        # Criterion
        self.loss = nn.BCELoss(weight=torch.tensor(
            class_weights, dtype=torch.float))

        # Metrics
        self.f1_metric = torchmetrics.F1(
            num_classes=n_classes, average="weighted")
        self.accuracy = torchmetrics.Accuracy(
            num_classes=n_classes, average='weighted')

        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels):
        output = self.bert(
            input_ids, attention_mask=attention_mask)

        sequence_output = output['last_hidden_state']

        output, hidden = self.gru(sequence_output)

        fc_input = self.dropout(F.relu(output[:, -1]))

        output = self.fc(fc_input)
        output = F.sigmoid(output)

        return output

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch.values()

        outputs = self(input_ids, attention_mask, labels)
        loss = self.loss(outputs, labels)
        labels = torch.argmax(labels, dim=1)

        self.log("loss/train", loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch.values()

        outputs = self(input_ids, attention_mask, labels)
        preds = torch.argmax(outputs, dim=1)
        loss = self.loss(outputs, labels)
        labels = torch.argmax(labels, dim=1)

        acc = self.accuracy(preds, labels)
        f1 = self.f1_metric(preds, labels)

        self.log("loss/val", loss, prog_bar=True, logger=True)
        self.log("acc/val", acc, prog_bar=True, logger=True)
        self.log("f1/val", f1, prog_bar=True, logger=True)

        return {"loss": loss, "acc": acc, "predictions": preds, "probs": outputs, "labels": labels}

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch.values()

        outputs = self(input_ids, attention_mask, labels)
        loss = self.loss(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        labels = torch.argmax(labels, dim=1)

        acc = self.accuracy(preds, labels)
        f1 = self.f1_metric(preds, labels)

        self.log("loss/test", loss, prog_bar=True, logger=True)
        self.log("acc/test", acc, prog_bar=True, logger=True)
        self.log("f1/test", f1, prog_bar=True, logger=True)

        return {"loss": loss, "acc": acc, "predictions": preds, "probs": outputs, "labels": labels}

    def test_epoch_end(self, outputs):
        labels = []
        predictions = []
        probs = []

        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)
            for out_probs in output["probs"].detach().cpu():
                probs.append(out_probs)

        labels = torch.stack(labels).numpy()
        predictions = torch.stack(predictions).numpy()
        probs = torch.stack(probs).numpy()

        y_true = labels

        if WANDB:
            # Log confusion matrix
            self.logger.experiment.log({"conf_mat/test": wandb.plot.confusion_matrix(
                preds=predictions, y_true=y_true, class_names=LABEL_COLUMNS)})
            # Log PR curve
            self.logger.experiment.log(
                {"pr/test": wandb.plot.pr_curve(y_true, probs, LABEL_COLUMNS)})
            # Log ROC curve
            self.logger.experiment.log(
                {"roc/test": wandb.plot.roc_curve(y_true, probs, LABEL_COLUMNS)})

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=LEARNING_RATE)
        return optimizer
