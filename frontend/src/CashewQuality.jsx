import React, { useState } from 'react';
import axios from 'axios';
import { Gauge } from '@mui/x-charts/Gauge';
import WarningAlert2 from './WarningAlert2';
import Loader from './Loader';

const CashewQuality = () => {
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
    <div className='text-white font-joan p-8 max-w-3xl mx-auto flex flex-col items-center justify-center gap-10'>
      <h1 className="text-5xl font-bold mb-4 mt-8">Cashew Quality Check</h1>

      <form onSubmit={handleSubmit} className="space-y-4 flex items-center justify-around w-full">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="block w-40 text-lg text-white mt-3"
        />

        <button
          type="submit"
          className="px-4 py-2 font-joan bg-black text-white border border-white rounded hover:bg-neutral-900"
          disabled={loading}
        >
          {loading ? 'Processing...' : 'Upload & Check'}
        </button>
      </form>

      {error && <div class="flex items-center justify-between max-w-80 w-full bg-red-600/20 text-red-600 px-3 h-10 rounded-sm">
    <div class="flex items-center">
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M10 14.167q.354 0 .593-.24.24-.24.24-.594a.8.8 0 0 0-.24-.593.8.8 0 0 0-.594-.24.8.8 0 0 0-.593.24.8.8 0 0 0-.24.593q0 .354.24.594t.593.24m-.834-3.334h1.667v-5H9.166zm.833 7.5a8.1 8.1 0 0 1-3.25-.656 8.4 8.4 0 0 1-2.645-1.781 8.4 8.4 0 0 1-1.782-2.646A8.1 8.1 0 0 1 1.666 10q0-1.73.656-3.25a8.4 8.4 0 0 1 1.782-2.646 8.4 8.4 0 0 1 2.645-1.781A8.1 8.1 0 0 1 10 1.667q1.73 0 3.25.656a8.4 8.4 0 0 1 2.646 1.781 8.4 8.4 0 0 1 1.781 2.646 8.1 8.1 0 0 1 .657 3.25 8.1 8.1 0 0 1-.657 3.25 8.4 8.4 0 0 1-1.78 2.646 8.4 8.4 0 0 1-2.647 1.781 8.1 8.1 0 0 1-3.25.656" fill="currentColor"/>
        </svg>
        <p class="text-lg ml-2 text-white">{error}</p>
    </div>   

</div>}
    {loading&&<Loader/>}
      {result && (
        <>
          <div className='mt-15 flex flex-col gap-10'>
            <h2 className="text-2xl text-center font-semibold text-white">
              Predicted Class: <span className='text-white text-3xl'>{result.predicted_class}</span>
            </h2>

            <div className="grid grid-cols-2 gap-20 text-white">
              <div>
                <img src={result.image_url} alt="Uploaded" className="rounded shadow h-50 w-50" />
                <h3 className="font-semibold mb-2 text-center mt-5">Uploaded Image</h3>
              </div>
              <div>
                <img src={result.heatmap_url} alt="Heatmap" className="rounded shadow h-50 w-50" />
                <h3 className="font-semibold mb-2 text-center mt-5">Heatmap</h3>
              </div>
            </div>
          </div>

          <div className='text-white mt-10 w-full'>
            <h2 className="text-xl font-semibold mb-6 text-center">Confidence Scores:</h2>
            <div className="flex flex-col gap-8">
              {Object.entries(result.confidence_scores).map(([label, score]) => (
                <div key={label} className="flex items-center justify-around">
                     <div className="relative w-[120px] h-[120px]">
                    <Gauge
                      value={score * 100}
                      startAngle={0}
                      endAngle={360}
                      innerRadius="80%"
                      outerRadius="100%"
                      width={120}
                      height={120}
                    />
                    <div className="absolute inset-0 flex items-center justify-center text-xl font-bold">
                      {(score * 100).toFixed(2)}%
                    </div>
                  </div>
                  <span className="text-lg w-32 text-center">{label}</span>
                 
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default CashewQuality;
