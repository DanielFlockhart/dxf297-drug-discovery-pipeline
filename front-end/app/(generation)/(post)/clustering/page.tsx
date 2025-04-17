"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";

export default function ClusteringPage() {
  const [plotUrl, setPlotUrl] = useState("/datasets/clustered.html");


  return (
    <div className="min-h-screen text-white p-8">
      <div className="text-center mb-12">
        <h1 className="text-5xl font-bold mb-4">Clustering Analysis</h1>
        <p className="text-lg text-gray-300">
          Analyze and visualize the clustering of your dataset.
        </p>
      </div>
      
      <iframe
          src={plotUrl}
          className="w-full h-[96vh]"
            ></iframe>

    </div>
  );
}
