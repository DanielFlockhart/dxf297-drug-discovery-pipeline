"use client";
export const dynamic = "force-dynamic"; // <= Tells Next.js not to SSG this page.
import { FIREBASE_APP } from "@/config/FirebaseConfig";

import React, { useState, useEffect } from "react";
import { collection, getDocs } from "firebase/firestore";
import { ref, getDownloadURL } from "firebase/storage";
import { effects } from "@/constants/effects";
import { FIREBASE_DB, FIREBASE_STORAGE } from "@/config/FirebaseConfig";
import { useRouter } from "next/navigation";
import { FaTrash } from "react-icons/fa";

export default function Setup() {
  const [currentStep, setCurrentStep] = useState(1);
  const [selectedEffects, setSelectedEffects] = useState({});
  const [molecules, setMolecules] = useState([]); // Full list of molecules
  const [filteredMolecules, setFilteredMolecules] = useState([]); // Molecules to display after filtering
  const [selectedMolecules, setSelectedMolecules] = useState([]);
  const [searchTerm, setSearchTerm] = useState(""); // Search input state
  const maxSelected = 5;
  const [loadingMolecules, setLoadingMolecules] = useState(true);
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  
  // Custom smiles variables
  const [customSmiles, setCustomSmiles] = useState("");
  const [customMolecules, setCustomMolecules] = useState([]);
  // Fetch molecules from Firestore
  const fetchMoleculeMetadata = async () => {
    const amount = 240; // Number of molecules to process
    const moleculesCollection = collection(FIREBASE_DB, "molecules");
  
    try {
      const querySnapshot = await getDocs(moleculesCollection);
      const molecules = querySnapshot.docs
        .slice(0, amount) // Limit the number of documents processed
        .map((doc) => ({
          id: doc.id,
          ...doc.data(),
        }));
  
      // Set metadata without images
      setMolecules(molecules);
      setFilteredMolecules(molecules);
      return molecules;
    } catch (error) {
      console.error("Error fetching molecule metadata:", error);
      setMolecules([]);
      setFilteredMolecules([]);
      return [];
    }
  };

  const fetchImageURLs = async (molecules) => {
    molecules.forEach((molecule) => {
      // Use setTimeout to allow incremental updates without blocking
      setTimeout(() => {
        try {
          const encodedPath = encodeURIComponent(molecule.img); // Handle encoding here
          const proxyURL = `/proxy-storage/${encodedPath}`; // Proxied URL
  
          // Update molecule with its image URL
          setMolecules((prevMolecules) =>
            prevMolecules.map((prev) =>
              prev.id === molecule.id ? { ...prev, imgURL: proxyURL } : prev
            )
          );
  
          setFilteredMolecules((prevFiltered) =>
            prevFiltered.map((prev) =>
              prev.id === molecule.id ? { ...prev, imgURL: proxyURL } : prev
            )
          );
        } catch (error) {
          console.error(`Error encoding image URL for molecule ${molecule.id}:`, error);
        }
      }, 0); // Defer execution to avoid blocking
    });
  
    // Final cleanup after initiating updates
    setLoadingMolecules(false);
  };
  

  const fetchMolecules = async () => {
    setLoadingMolecules(true);
  
    // Step 1: Fetch metadata
    const metadata = await fetchMoleculeMetadata();
  
    // Step 2: Fetch image URLs (only if metadata was successfully fetched)
    if (metadata.length > 0) {
      await fetchImageURLs(metadata);
    } else {
      setLoadingMolecules(false);
    }
  };
  
  
  

  useEffect(() => {
    fetchMolecules();
  }, []);
  useEffect(() => {
    if (loading) {
      console.log("Resetting...");
      setMolecules([]); // Optionally clear molecules if needed
      setFilteredMolecules([]);
    }
  }, [loading]);
  // Filter molecules whenever the search term changes
  useEffect(() => {
    const lowercasedSearchTerm = searchTerm.toLowerCase();
    const filtered = molecules.filter(
      (molecule) =>
        molecule.name.toLowerCase().includes(lowercasedSearchTerm) ||
        molecule.smile.toLowerCase().includes(lowercasedSearchTerm)
    );
    setFilteredMolecules(filtered);
  }, [searchTerm, molecules]);

  const hasSelectedEffects = Object.values(selectedEffects).some((isSelected) => isSelected);
  const hasSelectedMolecules = selectedMolecules.length > 0 || customMolecules.length > 0;

  const handleSelectEffect = (key) => {
    setSelectedEffects((prevSelected) => ({
      ...prevSelected,
      [key]: !prevSelected[key],
    }));
  };

  const handleSelectMolecule = (id) => {
    if (selectedMolecules.length >= maxSelected && !selectedMolecules.includes(id)) return;
    setSelectedMolecules((prevSelected) =>
      prevSelected.includes(id)
        ? prevSelected.filter((item) => item !== id)
        : [...prevSelected, id]
    );
  };
  const handleNavigateToGeneration = () => {
    setLoading(true);
    console.log("handling navigation to generation...");
  
    // Combine selected molecules with custom SMILES
    const combinedStartingMolecules = [
      ...selectedMolecules.map((id) => molecules.find((molecule) => molecule.id === id)),
      ...customMolecules.map((smile, index) => ({
        id: `custom-${index}`,
        name: null,
        smile: smile,
        imgURL: null, // No image URL available for custom SMILES
      })),
    ];
  
    const data = {
      startingMolecules: combinedStartingMolecules,
      desiredEffects: Object.entries(selectedEffects)
        .filter(([_, isSelected]) => isSelected)
        .map(([key]) => key),
    };
  
  
    const query = encodeURIComponent(JSON.stringify(data));
    console.log("Encoded data, navigating...");
    router.push(`/generation?data=${query}`);
  };

  // Step 1: Select Effects
  const stepOne = () => (
    <div className="flex flex-col gap-8 w-full max-w-7xl">
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-6xl font-bold mb-2 text-center">Select Effects</h1>
        <p className="text-lg text-gray-300">Decide which effects you want to target.</p>
      </div>
      <div className="flex items-center justify-center p-4 rounded-lg shadow-md backdrop-blur-md bg-gradient-to-br from-[rgba(0,0,0,0.2)] to-[rgba(255,255,255,0.1)]">
  <h2 className="text-2xl font-semibold text-center">Select Effects</h2>
</div>

      {/* Effects Selection */}
      <div className="text-black rounded-lg shadow-lg grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
          {Object.entries(effects).map(([key, effect]) => (
            <label
              key={key}
              className="flex items-center space-x-3 p-2  rounded-lg transition
              backdrop-blur-md hover:opacity-80 cursor-pointer 
      bg-gradient-to-br from-[rgba(0,0,0,0.2)] to-[rgba(255,255,255,0.1)]"
            >
              {/* Standardized Checkbox */}
              <input
                type="checkbox"
                className="form-checkbox h-5 w-5 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
                checked={!!selectedEffects[key]}
                onChange={() => handleSelectEffect(key)}
                style={{ minWidth: "1.25rem", minHeight: "1.25rem" }} // Explicit size
              />
              <span className="text-white">{effect}</span>
            </label>
          ))}
        </div>
  
      {/* Next Button */}
      <button
        onClick={() => hasSelectedEffects && setCurrentStep(2)}
        disabled={!hasSelectedEffects}
        className={`rounded-full transition-colors px-8 py-2 text-white font-semibold ${
          hasSelectedEffects ? "bg-blue-500 hover:bg-blue-600" : "bg-gray-300"
        }`}
      >
        Next
      </button>
    </div>
  );


  // Step 2: Select Molecules with Search Bar
  const stepTwo = () => (
    <div className="flex flex-col gap-6 w-full max-w-7xl">
      <div className="text-center mb-12">
        <h1 className="text-6xl font-bold mb-2">Select Starting Molecules</h1>
        <p className="text-lg text-gray-300">Choose the molecules you want to start with.</p>
      </div>

      <div className="flex items-center justify-center p-4 rounded-lg shadow-md backdrop-blur-md bg-gradient-to-br from-[rgba(0,0,0,0.2)] to-[rgba(255,255,255,0.1)]">
      <h2 className="text-xl font-semibold">
          Select Starting Molecules ({selectedMolecules.length}/{maxSelected})
        </h2>
      </div>

      <div className="flex flex-col items-center justify-center p-4 rounded-lg shadow-md backdrop-blur-md bg-gradient-to-br from-[rgba(0,0,0,0.2)] to-[rgba(255,255,255,0.1)]">
        <h2 className="text-xl font-semibold">
          Or Enter a custom smile
        </h2>
        <input
          type="text"
          value={customSmiles}
          onChange={(e) => setCustomSmiles(e.target.value)}

          placeholder="Enter a SMILE..."
          className="p-2 w-[50%] border border-gray-300 rounded-lg focus:outline-none focus:ring focus:ring-blue-300 mt-4 text-black"
        />
        {customMolecules.length >= 5 && (
          <p className="text-red-500 text-sm mt-2">You can only add up to 5 custom molecules.</p>
        )}
        <button
          onClick={() => {
            if (customSmiles && customMolecules.length < 5) {
              setCustomMolecules((prev) => [...prev, customSmiles]);
              setCustomSmiles(""); // clear input after adding
            }
          }}
          disabled={customMolecules.length >= 5 || !customSmiles.trim()}
          className={`rounded-full px-4 py-2 text-white mt-4 ${
            customMolecules.length >= 5 || !customSmiles.trim() ? "bg-gray-300" : "bg-blue-500"
          }`}
        >
          Add Molecule
        </button>

        <div className="flex flex-row flex-wrap justify-center w-full space-x-2 mt-4">
          {customMolecules.map((smile, index) => (
            <div key={index} className="space-x-2 flex items-center justify-between w-auto p-2 bg-gray-800 border-2 border-white rounded-full mt-4">
              <p className="text-white">
                {smile.length > 20 ? `${smile.substring(0, 20)}...` : smile}
              </p>

              <button
                onClick={() =>
                  setCustomMolecules((prev) =>
                    prev.filter((_, i) => i !== index)
                  )
                }
                className=""
              >
                <FaTrash color="red" />
              </button>
            </div>
          ))}
        </div>
      </div>
      
      <div className="text-black p-6 rounded-lg shadow-lg rounded-lg shadow-md backdrop-blur-md bg-gradient-to-br from-[rgba(0,0,0,0.2)] to-[rgba(255,255,255,0.1)]">

        {/* Search Bar */}
        <input
          type="text"
          placeholder="Search by name or SMILES..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="mb-4 p-2 w-full border border-gray-300 rounded-lg focus:outline-none focus:ring focus:ring-blue-300"
        />

        {!loadingMolecules ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
            {filteredMolecules.map((molecule) => (
              <div
                key={molecule.id}
                onClick={() => handleSelectMolecule(molecule.id)}
                className={`relative w-full h-full rounded shadow-md cursor-pointer overflow-hidden transition-transform transform hover:scale-105 ${
                  selectedMolecules.includes(molecule.id)
                    ? "border-blue-500 border-4"
                    : "border"
                }`}
              >
                {/* Image */}
                <img
                  src={molecule.imgURL || ""}
                  alt={molecule.name}
                  className="object-cover w-full aspect-square"
                />

                {/* Hover Overlay */}
                <div className="absolute inset-0 bg-black bg-opacity-60 text-white flex flex-col items-center justify-center opacity-0 hover:opacity-100 transition-opacity">
                  <p className="text-sm font-semibold">{molecule.name}</p>
                  <p className="text-xs">{molecule.smile}</p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div>Loading molecules...</div>
        )}
      </div>
      <div className="flex justify-between w-full mt-4">
        <button
          onClick={() => setCurrentStep(1)}
          className="rounded-full bg-gray-300 px-4 py-2"
        >
          Back
        </button>
        <button
          onClick={() => hasSelectedMolecules && setCurrentStep(3)}
          disabled={!hasSelectedMolecules}
          className={`rounded-full transition-colors ${
            hasSelectedMolecules ? "bg-blue-500" : "bg-gray-300"
          } px-8 py-2`}
        >
          Next
        </button>
      </div>
    </div>
  );

  const stepThree = () => (
    <div className="flex flex-col gap-8 w-full max-w-7xl">
      <div className="text-center mb-12">
        <h1 className="text-6xl font-bold mb-2">Review Selection</h1>
        <p className="text-lg text-gray-300">Check your selected effects, starting molecules, and custom SMILES.</p>
      </div>
  
      {/* Selected Effects */}
      <div className="text-black p-6 rounded-lg shadow-lg backdrop-blur-md bg-gradient-to-br from-[rgba(0,0,0,0.2)] to-[rgba(255,255,255,0.1)]">
        <h2 className="text-2xl font-semibold mb-4 text-white">Selected Effects</h2>
        <ul className="list-disc pl-5">
          {Object.entries(selectedEffects)
            .filter(([_, isSelected]) => isSelected)
            .map(([key]) => (
              <li key={key} className="text-white">
                {effects[key]}
              </li>
            ))}
        </ul>
      </div>
  
      {/* Selected Molecules */}
      <div className="text-black p-6 rounded-lg shadow-lg backdrop-blur-md bg-gradient-to-br from-[rgba(0,0,0,0.2)] to-[rgba(255,255,255,0.1)]">
        <h2 className="text-2xl font-semibold mb-4 text-white">Selected Starting Molecules</h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
          {selectedMolecules.map((id) => {
            const molecule = molecules.find((molecule) => molecule.id === id);
            return (
              <div key={id} className="relative w-full h-full overflow-hidden rounded shadow-md bg-gray-100">
                {molecule.imgURL && (
                  <img
                    src={molecule.imgURL}
                    alt={molecule.name}
                    className="object-cover w-full aspect-square"
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>
  
      {/* Custom SMILES Section */}
      {customMolecules.length > 0 && (
        <div className="text-black p-6 rounded-lg shadow-lg backdrop-blur-md bg-gradient-to-br from-[rgba(0,0,0,0.2)] to-[rgba(255,255,255,0.1)]">
          <h2 className="text-2xl font-semibold mb-4 text-white">Custom SMILES</h2>
          <ul className="space-y-2">
            {customMolecules.map((smile, index) => (
              <li key={index} className="text-white break-words">
                {smile.length > 50 ? `${smile.substring(0, 50)}...` : smile}
              </li>
            ))}
          </ul>
        </div>
      )}
  
      {/* Navigation Buttons */}
      <div className="flex justify-between w-full mt-4">
        <button
          onClick={() => setCurrentStep(2)}
          className="rounded-full bg-gray-300 px-4 py-2"
        >
          Back
        </button>
        <button
          onClick={handleNavigateToGeneration}
          className="rounded-full bg-blue-500 px-8 py-2 text-white"
        >
          Generate Molecules
        </button>
      </div>
    </div>
  );

  if (loading) {
    return (
      <div>
        <div className="relative min-h-screen flex flex-col items-center justify-center text-white p-8 sm:p-20 font-[family-name:var(--font-geist-sans)]">
          <h1 className="text-4xl font-semibold">Generating Molecules...</h1>
        </div>

      </div>
    )
  }

  return (
    <div>
      <div className="relative min-h-screen flex flex-col items-center justify-center text-white p-8 sm:p-20 font-[family-name:var(--font-geist-sans)]">
        <main className="flex flex-col gap-12 items-center relative z-10 w-full">
          {currentStep === 1 && stepOne()}
          {currentStep === 2 && stepTwo()}
          {currentStep === 3 && stepThree()}
        </main>
      </div>
    </div>
  );
}
