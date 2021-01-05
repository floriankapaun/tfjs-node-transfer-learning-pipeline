const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const mobilenet = require('@tensorflow-models/mobilenet');
const knnClassifier = require('@tensorflow-models/knn-classifier');

const data = require('./data/index.js');

const KNN_MODEL_PATH = './knn-model.js';

const model = mobilenet.load();
const classifier = knnClassifier.create();


/**
 * Train the KNN classifier with (1) class and (2) activation of MobileNet
 * feeded an image.
 * 
 * @param {String|Integer} classId - image label
 * @param {String} imageAbsolutePath - image to train the classifier with
 * 
 * @returns {Array}
 */
const train = async (classId, imageAbsolutePath) => {
    // load image
    const image = fs.readFileSync(imageAbsolutePath);
    const decodedImage = tf.node.decodeImage(image, 3);
    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier. This procedure is know as "transfer learning".
    const activation = (await model).infer(decodedImage, true);
    classifier.addExample(activation, classId);
    return [classId, imageAbsolutePath];
}


/**
 * Stringifies the KNN Model dataset
 * 
 * @param {Object} dataset - retrieved from classifier.getClassifierDataset()
 * 
 * @returns {String}
 */
const stringify = (dataset) => {
    let data = Object.entries(dataset);
    data = data.map(([label, data]) => [label, Array.from(data.dataSync()), data.shape]);
    return JSON.stringify(data);
}


/**
 * Save the KNN Model to the file system
 * 
 * @param {String} absolutePath – path including name for export
 */
const save = async (absolutePath) => {
    try {
        // Get dataset from classifier
        const dataset = classifier.getClassifierDataset();
        // Stringify the dataset
        const data = stringify(dataset);
        // Write to file system
        await fs.writeFile(absolutePath, data, (error) => { 
            if (error) {
                throw(error); 
            } else { 
                console.log(`Successfully exported KNN Model to ${absolutePath}`); 
            } 
        });
    } catch (error) {
        console.error(error);
    }
}

// Train the KNN Classifier with examples found in data.train
const examples = [];
for (const example of data.train) {
    const promise = train(example.class, example.data);
    // Save all promises in an array
    examples.push(promise);
}

// If all examples are added and their according promises resolved
// export the KNN Model to the file system.
Promise.all(examples).then(() => {
    save(KNN_MODEL_PATH);
});