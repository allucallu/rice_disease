import * as tf from '@tensorflow/tfjs';
import { useState, useEffect } from 'react';

export default function Home() {
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState(null);

  // Load model saat komponen mount
  useEffect(() => {
    async function loadModel() {
      const loadedModel = await tf.loadLayersModel('/tfjs_model/model.json');
      setModel(loadedModel);
    }
    loadModel();
  }, []);

  // Prediksi gambar
  const handlePredict = async (e) => {
    const file = e.target.files[0];
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);

    img.onload = async () => {
      // Preprocess gambar
      const tensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(255.0)
        .expandDims();

      // Prediksi
      const preds = await model.predict(tensor).data();
      const classNames = ['leaf_blast', 'healthy', 'brown_spot', 'bacterial_leaf_blight']; // Ganti dengan nama kelas Anda
      const predictedClass = classNames[preds.indexOf(Math.max(...preds))];
      setPrediction(`Predicted: ${predictedClass}`);
    };
  };

  return (
    <div>
      <h1>Plant Disease Classifier</h1>
      <input type="file" accept="image/*" onChange={handlePredict} />
      {prediction && <p>{prediction}</p>}
    </div>
  );
}