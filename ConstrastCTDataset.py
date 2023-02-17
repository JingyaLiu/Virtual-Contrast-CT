class ConstrastCTDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_dir, label_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(get_id(self.image_dir))

    def get_ids(dir):
        """Returns a list of the ids in the directory"""
        return (f for f in os.listdir(dir))
    
    def __getitem__(self, idx):
        print('len',  len(get_ids(image_dir)[idx]))
        img_name = os.path.join(self.image_dir,
                                get_ids(image_dir)[idx])
        image = io.imread(img_name)
        label_name = os.path.join(self.label_dir,
                                get_ids(label_dir)[idx])
        label = io.imread(label_name)
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
