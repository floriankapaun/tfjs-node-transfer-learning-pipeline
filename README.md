# tfjs Emotion Classification

tfjs Emotion Classification is a [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) application for classifying images according to human emotions based on [Tensorflow.js](https://github.com/tensorflow/tfjs).

It's using a [KNN Classifier](https://github.com/tensorflow/tfjs-models/tree/master/knn-classifier) on top of [Mobilenet](https://github.com/tensorflow/tfjs-models/tree/master/mobilenet).

## Instructions

I'm using [yarn](https://classic.yarnpkg.com/en/docs/install) in my examples but you could use [npm](https://www.npmjs.com/) as well.

1. Get all dependencies.

    ```yarn install```

2. Train the model. This will use input data from `./data/train` and export the trained model to `./knn-model.js`. You can simply replace the data in those folders and adjust the `./data/index.js` to your needs.

    ```yarn train```

3. Test the trained model. This will import the trained model and test it with data from `./data/test`. Wrong classifications will be printed in the console.

    ```yarn test```

## Credits

Used sample data is taken from this [repository](https://github.com/swimauger/image-classifier) by [swimaugers](https://github.com/swimauger).