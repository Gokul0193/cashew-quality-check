import React, { useState } from 'react';
import axios from 'axios';

const CashewQualityCheck = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null); // clear previous result
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a file');
      return;
    }
    setLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await axios.post('http://127.0.0.1:5000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(res.data);
      console.log(res.data);
      
    } catch (err) {
      console.error(err);
      setError('Error connecting to server. Make sure Flask is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 max-w-xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Cashew Quality Check</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="block w-full text-sm text-gray-500"
        />
        <button
          type="submit"
          className="px-4 py-2 font-joan bg-green-600 text-white rounded hover:bg-green-700"
          disabled={loading}
        >
          {loading ? 'Processing...' : 'Upload & Check'}
        </button>
      </form>

      {error && <div className="mt-4 text-red-600">{error}</div>}

      {result && (
        <div className="mt-6 space-y-4">
          <div>
            <h2 className="text-xl font-semibold">Predicted Class:</h2>
            <p className="text-lg">{result.predicted_class}</p>
          </div>

          <div>
            <h2 className="text-xl font-semibold">Confidence Scores:</h2>
            <ul className="list-disc ml-6">
              {Object.entries(result.confidence_scores).map(([label, score]) => (
                <li key={label}>
                  {label}: {(score * 100).toFixed(2)}%
                </li>
              ))}
            </ul>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <h3 className="font-semibold mb-2">Uploaded Image</h3>
              <img src={result.image_url} alt="Uploaded" className="rounded shadow" />
            </div>
            <div>
              <h3 className="font-semibold mb-2">Heatmap</h3>
              <img src={result.heatmap_url} alt="Heatmap" className="rounded shadow" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CashewQualityCheck;
