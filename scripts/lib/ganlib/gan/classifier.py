import torch
from absl import flags, app
from absl.flags import FLAGS
import numpy as np
import torch.nn as nn
#from torch import max, mean, save, tensor, no_grad, where
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from gan.getData import getData
from gan.utils import unfreeze
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

class Vectorize(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBn2d, self).__init__()
        self.convbn2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.convbn2d(x)

class ConvBlock(nn.Module):
    def __init__(self, inter_channels, intra_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inter_channels, intra_channels, 3, 1, 1),
            nn.BatchNorm2d(intra_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(intra_channels, inter_channels, 3, 1, 1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x_in = x
        x = self.conv1(x)
        x = self.conv2(x)
        out = x_in + x
        return out

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.build()
        self._init_weights()

    def _init_weights(self):
        for i in self.conv.modules():
            if not isinstance(i, nn.modules.container.Sequential):
                classname = i.__class__.__name__
                if hasattr(i, "weight"):
                    if classname.find("Conv") != -1:
                        nn.init.xavier_uniform_(i.weight)
                    elif classname.find("BatchNorm2d") != -1:
                        nn.init.normal_(i.weight.data, 1.0, 0.02)
                if hasattr(i, 'bias'):
                    nn.init.zeros_(i.bias)
                    if classname.find("BatchNorm2d") != -1:
                        nn.init.constant_(i.bias.data, 0.0)

    def build(self):
        self.conv = nn.Sequential(
            # (batch_sz X 1 X 256 X 256)
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.MaxPool2d(2,2),
            # (batch_sz X 16 X 128 X 128)
            ConvBlock(16, 32),
            nn.MaxPool2d(2, 2),
            ConvBn2d(16, 32),
            # (batch_sz X 32 X 64 X 64)
            ConvBlock(32, 64),
            nn.MaxPool2d(2, 2),
            ConvBn2d(32, 64),
            # (batch_sz X 64 X 32 X 32)
            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2),
            ConvBn2d(64, 128)
            # (batch_sz X 128 X 16 X 16)
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, 16),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        out = self.conv(x)
        out = torch.mean(out, dim=-1)
        out, _ = torch.max(out, dim=-1)
        out = self.fc(out)
        return out

class ClfDataset(Dataset):
    def __init__(self, distorted_data, syn_ideal):
        super(ClfDataset, self).__init__()
        # TODO: `distorted_data` contains images of `real_distorted`.
        self.distorted_data = np.array(distorted_data)
        self.syn_ideal = np.array(syn_ideal)
        self.dataset = np.concatenate([self.syn_ideal, self.distorted_data])
        print(self.distorted_data.shape)
        print(self.syn_ideal.shape)
        print(self.dataset.shape)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        pos = np.random.random_integers(0, len(self.dataset)-1)
        #print(pos)
        if pos < len(self.syn_ideal):
            cls = [1.]
        else:
            cls = [0.]

        sample = {'data': torch.tensor(self.dataset[pos]).to(torch.float32),
                  'class': torch.tensor(cls).to(torch.float32)}
        return sample

def plot_roc_curve(fpr, tpr):
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr,tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) CURVE")
    plt.savefig("./trained_models/roc_curve.png")

def train(batch_size, n_epochs=30, lr=1e-3, weight_decay=0.14, workers=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Define the model
    clf = Classifier()
    clf.to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    clf = unfreeze(clf)
    optimizer = Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
    criterion.to(device)
    distorted_data, syn_ideal, train_real_distorted, test_real_distorted = getData(use_real=True)
    ldd = distorted_data.shape[0]
    lsi = syn_ideal.shape[0]
    traindataset = ClfDataset(distorted_data=distorted_data[:int(0.8*ldd)], syn_ideal=syn_ideal[:int(0.8*lsi)])
    testdataset = ClfDataset(distorted_data=distorted_data[int(0.8*ldd):], syn_ideal=syn_ideal[int(0.8*lsi):])
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    for epoch in range(n_epochs):
        train_loss = 0
        val_loss = 0
        matrix_total = 0
        predictions = np.array([])
        ground_truths = np.array([])
        with tqdm(trainloader, unit="batch") as tepoch:
            for i, batch in enumerate(tepoch):
                sample = batch['data']
                y_true = batch['class']
                # zero gradients
                optimizer.zero_grad()
                y_pred = clf(sample.to(device))
                loss = criterion(y_pred, y_true)
                y_pred = torch.sigmoid(y_pred)
                y_pred = y_pred.where(y_pred<0.5, torch.ones(y_pred.shape))
                y_pred = y_pred.where(y_pred>=0.5, torch.zeros(y_pred.shape))
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred.detach().numpy()).ravel()
                acc = tp/(tp+fp)
                rec = tp/(tp+fn)
                # Calculate gradients
                loss.backward()
                # Update weights
                optimizer.step()
                # accumulate train loss
                train_loss += loss.item()
                tepoch.set_postfix_str(f"[{epoch}/{n_epochs}][{i}/{len(trainloader)}]: loss={loss:.2f}, accuracy={acc:.2f}, recall={rec:.2f}")


    with torch.no_grad():
        with tqdm(testloader, unit="batch") as tepoch:
            for i, batch in enumerate(tepoch):
                sample = batch['data']
                y_true = batch['class']
                y_pred = clf(sample.to(device))
                y_pred = torch.sigmoid(y_pred)
                y_pred = torch.where(y_pred>.5, 1., 0.)
                loss = criterion(y_pred, y_true)
                val_loss += loss.item()
                matrix = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = matrix.ravel()
                acc = np.nan_to_num(tp / (tp + fp))
                rec = np.nan_to_num(tp / (tp + fn))
                matrix_total += matrix
                predictions = np.concatenate([predictions, y_pred.numpy().reshape((-1))])
                ground_truths = np.concatenate([ground_truths, y_true.numpy().reshape((-1))])
                tepoch.set_postfix_str(f"[{i}/{len(testloader)}]: loss={loss:.2f}, accuracy={acc:.2f}, recall={rec:.2f}")


    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    predictions = predictions.reshape((-1,1))
    ground_truths = ground_truths.reshape((-1,1))
    fpr, tpr, _ = roc_curve(ground_truths, predictions, pos_label=1)
    plot_roc_curve(fpr, tpr)
    torch.save(clf.state_dict(), './trained_models/Classifier.pt')

def main(argv):
    train(batch_size=FLAGS.batch_size, n_epochs=FLAGS.n_epochs, lr=FLAGS.learning_rate, workers=1)


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    app.run(main)