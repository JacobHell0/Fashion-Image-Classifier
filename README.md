# Fashion-Image-Classifier

## YouTube video explaining the final project
[Video Link](https://youtu.be/WIcqo7YlLhA?si=_-ei0hbfSTB9Hj9M)

## Project Description
This project was done for a Machine Learning course where we were tasked choosing a dataset and coming up with a well formulated problem that can be addressed with a machine learning solution.

## Dataset Link
[Link To Kaggle Fashion Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

## My Contribution
Everyone in the group did a fantastic job working on the project. My main contribution to the project was building the basic Convolutional Neural Network and having it be able to read the dataset.


### Specifics of My Convolutional Neural Network (CNN)

Here is layers of the CNN according to ```torchinfo.summary()```
![image](https://github.com/user-attachments/assets/410964c5-32a4-483f-91a3-697bfd214c1e)

I also wrote the training and evaluation loop for the CNN, for more information on this, I encourage you to look at the notebook (.ipynb) file, or have a look at the pdf output.

Here is the code I wrote to create the CNN model

```Python
class BasicConvModel(LightningModule):
  def __init__(self, num_labels, image_size_x=60, image_size_y=80):
    super().__init__()
    self.num_labels = num_labels
    self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=num_labels)

    self.linear_layer_size = 256 * (image_size_x // 4) * (image_size_y // 4)

    self.model = nn.Sequential(
          # First layer
          nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),  # 3 channel -> 16 channels
          nn.MaxPool2d(kernel_size=2), #image size: 60x80 -> 30x40
          nn.ReLU(),

          # Second conv layer
          nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),  # 16 channels -> 32 channels
          nn.MaxPool2d(kernel_size=2), #image size: 30x40 -> 15x20
          nn.ReLU(),

          # Flatten layer
          nn.Flatten(),  # 256*15*20 -> 76 800

          # Linear layer
          nn.Linear(self.linear_layer_size, self.num_labels)
      )

  def configure_optimizers(self):
      return torch.optim.Adam(self.parameters())

  def shared_step(self, mode:str, batch:Tuple[Tensor, Tensor], batch_index:int):
      x, target = batch
      output = self.forward(x)
      loss = self.loss(output, target)
      self.accuracy(output, target)
      self.log(f"{mode}_step_acc", self.accuracy, prog_bar=True)
      self.log(f"{mode}_step_loss", loss, prog_bar=False)
      return loss

  def training_step(self, batch, batch_index):
      return self.shared_step('train', batch, batch_index)

  def validation_step(self, batch, batch_index):
      return self.shared_step('val', batch, batch_index)

  def test_step(self, batch, batch_index):
      return self.shared_step('test', batch, batch_index)

  def forward(self, x):
      return self.model(x)

  def loss(self, logits, target):
        return nn.functional.cross_entropy(logits, target)
```
