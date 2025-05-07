import * as tf from '@tensorflow/tfjs';
import { useState, useEffect } from 'react';

export default function Home() {
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState('');

  useEffect(() => {
    async function loadModel() {
        try {
          // Opsi 1: Coba muat sebagai Layers Model
          let model = await tf.loadLayersModel('/tfjs_model/model.json');
          
          // Jika masih error, buat wrapper dengan input shape manual
          if (!model.inputs[0].shape) {
            console.warn('Model has no input shape, applying manual fix...');
            const input = tf.input({shape: [224, 224, 3]});
            model = tf.model({
              inputs: input,
              outputs: model.apply(input)
            });
          }
          
          return model;
        } catch (error) {
          console.error('Layers model failed:', error);
          
          // Opsi 2: Coba muat sebagai Graph Model
          try {
            const model = await tf.loadGraphModel('/tfjs_model/model.json');
            console.log('Loaded as GraphModel');
            return model;
          } catch (graphError) {
            throw new Error(`Both formats failed: ${graphError}`);
          }
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
