# tfjs Emotion Classification

tfjs Emotion Classification is a [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) application for classifying images according to human emotions based on [Tensorflow.js](https://github.com/tensorflow/tfjs).

It's using a [KNN Classifier](https://github.com/tensorflow/tfjs-models/tree/master/knn-classifier) on top of [Mobilenet](https://github.com/tensorflow/tfjs-models/tree/master/mobilenet).

## Instructions

1. Get all dependencies

    ```yarn install```

2. Train the model. This will take input data from `./data` and export the trained model to `./knn-model.js`.

    ```yarn train```

3. Test the trained model. This will import the trained model and print wrong classifications to your console.

    ```yarn test```

## Credits

Used sample data is taken from this [repository](https://github.com/swimauger/image-classifier) by [swimaugers](https://github.com/swimauger).