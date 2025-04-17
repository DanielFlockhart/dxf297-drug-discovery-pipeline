"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";

// Configuration for individual molecule analysis
const MOLECULE_TECHNIQUES = [
  {
    id: "profile",
    title: "Get Molecule Profile",
    description: "View detailed information about the molecule, including classes and effects.",
    gradient: "from-blue-500 to-green-500 hover:from-blue-600 hover:to-green-600",
    url: "/molecule-overview",
  },
  {
    id: "synthesis",
    title: "Generate Synthesis Route",
    description: "Create a synthesis route for the molecule to understand its production process.",
    gradient: "from-yellow-500 to-orange-500 hover:from-yellow-600 hover:to-orange-600",
    url: "/molecule-synthesis",
  },
  {
    id: "dataset",
    title: "Visualise Dataset",
    description: "Visaulise the dataset to identify patterns and relationships.",
    gradient: "from-red-500 to-purple-500 hover:from-red-600 hover:to-purple-600",
    url: "/clustering",
  },
];


export default function Analytics() {
  const [step, setStep] = useState("select");
  const [selectedOption, setSelectedOption] = useState(null); 
  const [smiles, setSmiles] = useState(""); 
  const router = useRouter();


  const handleOptionSelect = (option) => {
    setSelectedOption(option);
    if (option.id === "dataset") {
      handleSubmit(true);
    }
    setStep("input");
  };

  const handleSubmit = (isTheDataset = false) => {
    const targetUrl = isTheDataset
      ? `${selectedOption.url}`
      : `${selectedOption.url}?smiles=${encodeURIComponent(smiles)}`;
    console.log("Navigating to:", targetUrl);
  
    router.push(targetUrl);
  };
  

  return (
    <div className="min-h-screen text-white p-8 flex flex-col items-center justify-center">
      {step === "select" && (
        <>
          <div className="text-center mb-12">
            <h1 className="text-5xl font-bold mb-4">Analytics</h1>
            <p className="text-lg text-gray-300">
              Select an option to analyze an individual molecule or a dataset.
            </p>
          </div>
          <div className="w-full max-w-6xl mb-12">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
              {MOLECULE_TECHNIQUES.map((technique) => (
                <div
                  key={technique.id}
                  className={`bg-gradient-to-br ${technique.gradient} p-6 rounded-xl shadow-lg cursor-pointer transition transform hover:scale-105 flex flex-col justify-center items-center text-center`}
                  onClick={() => handleOptionSelect(technique)}
                >
                  <h2 className="text-2xl font-semibold mb-4">{technique.title}</h2>
                  <p className="text-gray-200">{technique.description}</p>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {step === "input" && (
        <>
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold mb-4">
              {selectedOption.id === "dataset" ? "Analyze Dataset" : "Analyze Molecule"}
            </h1>
            <p className="text-lg text-gray-300">
              Enter the SMILES string for the molecule you want to analyze.
            </p>
            {selectedOption.id  === "synthesis" && 
              <p className="text-lg text-red-500 max-w-md">
              When synthesising some molecules, the natural language model will occasionally reject certain molecules on ethical grounds.
            </p>}
          </div>
          
          <div className="flex flex-col items-center space-y-6 w-full max-w-md">
            
              <input
                type="text"
                value={smiles}
                onChange={(e) => setSmiles(e.target.value)}
                placeholder="Enter SMILES string"
                className="w-full p-4 rounded-lg bg-gray-800 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              />

            <button
              onClick={() => handleSubmit(false)}
              className="w-full p-4 rounded-lg bg-gradient-to-r from-blue-500 to-green-500 hover:from-blue-600 hover:to-green-600 text-white text-lg font-semibold shadow-lg transition transform hover:scale-105"
            >
              Confirm
            </button>

            <button
              onClick={() => setStep("select")}
              className="w-full p-4 rounded-lg bg-gray-700 text-gray-200 hover:bg-gray-600 text-lg font-semibold shadow-lg transition transform hover:scale-105"
            >
              Go Back
            </button>
          </div>
        </>
      )}
    </div>
  );
}
