import * as tf from '@tensorflow/tfjs';
import { useState, useEffect } from 'react';

export default function Home() {
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState('');

  useEffect(() => {
    async function loadModel() {
      try {
        const loadedModel = await tf.loadLayersModel('/tfjs_model/model.json');
        
        // Workaround untuk model tanpa input shape
        if (!loadedModel.inputs[0].shape) {
          const input = tf.input({ shape: [224, 224, 3] });
          const newModel = tf.model({ 
            inputs: input, 
            outputs: loadedModel.apply(input) 
          });
          setModel(newModel);
        } else {
          setModel(loadedModel);
        }
      } catch (error) {
        console.error("Failed to load model:", error);
      }
    }
    loadModel();
  }, []);

  const handlePredict = async (e) => {
    const file = e.target.files[0];
    if (!file || !model) return;

    const img = new Image();
    img.src = URL.createObjectURL(file);

    img.onload = async () => {
      const tensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(255.0)
        .expandDims();

      const preds = await model.predict(tensor).data();
      const classNames = ['leaf_blast', 'healthy', 'brown_spot', 'bacterial_leaf_blight']; // Sesuaikan
      const topPrediction = {
        class: classNames[preds.indexOf(Math.max(...preds))],
        confidence: Math.max(...preds) * 100
      };
      setPrediction(`Predicted: ${topPrediction.class} (${topPrediction.confidence.toFixed(2)}%)`);
      tf.dispose(tensor);
    };
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>Plant Disease Classifier</h1>
      <input type="file" accept="image/*" onChange={handlePredict} />
      {prediction && <p style={{ marginTop: '20px', fontSize: '18px' }}>{prediction}</p>}
    </div>
  );
}