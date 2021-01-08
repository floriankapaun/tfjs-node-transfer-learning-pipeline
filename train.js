const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const mobilenet = require('@tensorflow-models/mobilenet');
const knnClassifier = require('@tensorflow-models/knn-classifier');

const data = require('./data/index.js');

const KNN_MODEL_PATH = './knn-model.js';
const MAX_IMAGES_PER_CATEGORY = 150;
const FILE_FORMAT_REGEX = new RegExp('\.(jp(e?)g|png)$');

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
try {
    for (const classificationClass of data.train) {
        // Get image fileNames for each class
        const fileNames = fs.readdirSync(classificationClass.data);
        // For each image
        for (let i = 0; i < MAX_IMAGES_PER_CATEGORY && i < fileNames.length ; i++) {
            // Make sure the file format is correct/supported
            if (FILE_FORMAT_REGEX.test(fileNames[i])) {
                // Train the KNN Classifier with that image
                const promise = train(classificationClass.id, classificationClass.data + fileNames[i]);
                // Save all promises in an array
                examples.push(promise);
            }
        }
    }
} catch (error) {
    console.error(error);
}

// If all examples are added and their according promises resolved
// export the KNN Model to the file system.
Promise.all(examples).then(() => {
    save(KNN_MODEL_PATH);
});