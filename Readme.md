# Face-Detection-Recognition

This project presents a full process of creating face-detection & face-recognition system.

# Dataset

The dataset should a directory of images that should be like follows:

```
├── dataset
│   ├──people
│   │   ├── person-1
│   │   │  ├──images
│   │   │  │  ├──image-name-1.jpg
│   │   │  │  ├──image-name-2.jpg
│   │   │  │  ├──...
│   │   ├── person-2
│   │   │  ├──images
│   │   │  │  ├──image-name-1.jpg
│   │   │  │  ├──image-name-2.jpg
│   │   │  │  ├──...
│   │   ├──...
...
```

Notes:

1. person-name should be the name or ssn-id of the people that will be recognized.
2. image-names are arbitrary and can be anything.

## Preparation:

run the following code to prepare the dataset

```commandline
python data_preparation.py --dataset_dir dataset/people
```

By running the above code, faces of images will be cropped and saved in `cropped` folders under `people/person-names`.
In addition, the encoding of each person is extracted and saved in `people/encodings`. Finally, a single file called
`people.pkl` is created in `people` directory which contains a dictionary of names and encodings which will, eventually,
used to recognize persons in a video. The output should be like the following picture.
![](images/dataset_directory.png)