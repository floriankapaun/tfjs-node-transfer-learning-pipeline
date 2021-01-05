const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const mobilenet = require('@tensorflow-models/mobilenet');
const knnClassifier = require('@tensorflow-models/knn-classifier');

const data = require('./data/index.js');

const KNN_MODEL_PATH = './knn-model.js';

const model = mobilenet.load();
const classifier = knnClassifier.create();


/**
 * Classify an image
 * 
 * @param {String} imageAbsolutePath - image to classify
 * 
 * @returns {Object} prediction
 */
const classify = async (imageAbsolutePath) => {
    // load image
    const image = fs.readFileSync(imageAbsolutePath);
    const decodedImage = tf.node.decodeImage(image, 3);
    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier. This procedure is know as "transfer learning".
    const activation = (await model).infer(decodedImage, true);
    // Let the KNN classifier predict to which class belongs the image from which 
    // the activation originates.
    const prediction = await classifier.predictClass(activation);
    return prediction;
}


/**
 * Parses stringified KNN Model dataset
 * 
 * @param {Object} dataset - retrieved from classifier.getClassifierDataset()
 * 
 * @returns {Object}
 */
const parse = (data) => {
    let dataset = JSON.parse(data);
    dataset = dataset.map(([label, data, shape]) => [label, tf.tensor(data, shape)]);
    return Object.fromEntries(dataset);
}


/**
 * Load KNN Model from the file system
 * 
 * @param {String} absolutePath – path including name for export
 * 
 * @returns {Object}
 */
const load = async (absolutePath) => {
    
    try {
        if (fs.existsSync(absolutePath)) {
            // Load classifier dataset
            const data = new Promise((resolve, reject) => {
                fs.readFile(absolutePath, 'utf8', function (error, data) {
                    if (error) reject(error);
                    resolve(data);
                });
            });
            const dataset = parse((await data));
            // Apply it to KNN classifier
            classifier.setClassifierDataset(dataset);
            return classifier;
        } else {
            throw `File does not exist: ${absolutePath}`;
        }
    } catch (error) {
        console.error(error);
    }
}


/**
 * Test the loaded KNN model by letting it classify test data
 */
const test = async () => {
    // Load the KNN classifier model
    await load(KNN_MODEL_PATH);
    // Test the KNN Classifier with examples found in data.test
    for (const example of data.test) {
        const prediction = await classify(example.data);
        if (prediction.label !== example.class) {
            // Console log wrong predictions
            console.log('\x1b[33m', `Wrong prediction for image "${example.data}" \n Prediction: "${prediction.label}"\n Confidences: "${JSON.stringify(prediction.confidences)}" \n Reality: "${example.class}"`, '\x1b[0m');
        }
    }
}

test();