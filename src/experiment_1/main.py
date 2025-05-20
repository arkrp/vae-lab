import logging
import os
def main():
    logging.basicConfig(filename=".log", filemode='w', level=logging.INFO)
    import Model.NonUniformEncoder
    import Model.Decoder
    from torchvision import datasets
    from torchvision.transforms import ToTensor
    training_data = datasets.MNIST(
        root=os.environ['DATA_DIR'],
        train=True,
        download=True,
        transform=ToTensor())
    testing_data = datasets.MNIST(
        root=os.environ['DATA_DIR'],
        train=False,
        download=True,
        transform=ToTensor())
if __name__=="__main__":
    main()
