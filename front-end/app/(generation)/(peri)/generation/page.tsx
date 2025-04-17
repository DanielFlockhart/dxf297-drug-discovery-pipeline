"use client";

import { useRouter, useSearchParams } from "next/navigation";
import React, { Suspense } from "react";
import { effects } from "@/constants/effects";
import { FIREBASE_AUTH } from '@/config/FirebaseConfig';

export const dynamic = "force-dynamic";

export default function GenerationPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <GenerationPageContent />
    </Suspense>
  );
}

function GenerationPageContent() {
  const [results, setResults] = React.useState([]);           // [{ images, smiles }]
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState(null);
  const [expandedSections, setExpandedSections] = React.useState({});
  const [loadingMoleculeGens, setLoadingMoleculeGens] = React.useState({});
  const searchParams = useSearchParams();
  const router = useRouter();
  const generatedPerMolecule = 10;

  // parse input data
  const [startingMolecules, setStartingMolecules] = React.useState([]);
  const [desiredEffects, setDesiredEffects] = React.useState([]);

  React.useEffect(() => {
    const data = searchParams.get("data");
    if (data) {
      try {
        const decoded = JSON.parse(decodeURIComponent(data));
        setStartingMolecules(decoded.startingMolecules);
        setDesiredEffects(decoded.desiredEffects);

        // default all expanded
        const initExp = {};
        decoded.startingMolecules.forEach((_, i) => { initExp[i] = true });
        setExpandedSections(initExp);
      } catch (e) {
        console.error("Error parsing query data:", e);
        setError("Invalid input data");
      }
    }
    setLoading(false);
  }, [searchParams]);

  React.useEffect(() => {
    if (!startingMolecules.length || !desiredEffects.length) return;

    const generateAll = async () => {
      setLoading(true);
      setError(null);

      // build effects vector
      const numEffects = Object.keys(effects).length;
      const effectsArray = Array.from({ length: numEffects }, (_, i) =>
        desiredEffects.includes(String(i)) ? 1 : 0
      );

      const url = "https://erudite-backend-798229031686.europe-west2.run.app/generateMolecules";
      const allResults = [];

      for (let idx = 0; idx < startingMolecules.length; idx++) {
        setLoadingMoleculeGens(prev => ({ ...prev, [idx]: true }));
        const smile = startingMolecules[idx].smile;

        try {
          const res = await fetch(url, {
            method: "POST",
            headers: { 
              "Content-Type": "application/json",
              "Accept": "application/json"
            },
            body: JSON.stringify({
              smiles: smile,
              desired_effect: effectsArray,
              generate_molecules: generatedPerMolecule,
            }),
          });

          if (!res.ok) {
            console.error(`Failed at index ${idx}: ${res.statusText}`);
            allResults[idx] = { images: [], smiles: [] };
          } else {
            const data = await res.json();
            allResults[idx] = {
              images: data.images,
              smiles: data.molecules,
            };
          }
        } catch (err) {
          console.error(`Error generating molecules for ${smile}:`, err);
          allResults[idx] = { images: [], smiles: [] };
        } finally {
          setResults([...allResults]);  // update on each iteration
          setLoadingMoleculeGens(prev => ({ ...prev, [idx]: false }));
        }
      }

      setLoading(false);
    };

    generateAll();
  }, [startingMolecules, desiredEffects]);

  const toggleSection = (i) =>
    setExpandedSections(prev => ({ ...prev, [i]: !prev[i] }));

  const handleClick = (smile) =>
    router.push(`/molecule-overview?smiles=${encodeURIComponent(smile)}`);

  if (loading) {
    return (
      <div className="relative min-h-screen flex items-center justify-center flex-col">
        
        <h1 className="text-3xl font-semibold mb-6">Generating Molecules..</h1>
        <p className="text-lg text-gray-300">
          This may take some time depending on the number of molecules and effects.
        </p>
      </div>
    );
  }
  if (error) {
    return (
      <div className="relative min-h-screen flex items-center justify-center text-center">
        <h1 className="text-3xl font-semibold mb-6 text-center">Error</h1>
        <div className="text-red-500 text-center">{error}</div>
      </div>
    );
  }

  return (
    <div className="relative min-h-screen flex flex-col items-center justify-center text-white p-8 sm:p-20">
      <div className="w-full max-w-6xl p-8">
        <div className="text-center mb-12">
          <h1 className="text-6xl font-bold mb-2">Results</h1>
          <p className="text-lg text-gray-300">
            Here are the molecules generated based on your desired effects and starting molecules
          </p>
        </div>
        <h2 className="text-xl font-semibold mb-4 text-center text-gray-200">
          Desired Effects:{" "}
          {desiredEffects
            .slice(0, 5)
            .map((e) => effects[e])
            .join(", ")}
          {desiredEffects.length > 5 &&
            ` and ${desiredEffects.length - 5} more`}
        </h2>

        {startingMolecules.map((startingMolecule, moleculeIndex) => {
          const result = results[moleculeIndex] || {
            images: [],
            smiles: [],
          };

          return (
            <div key={moleculeIndex} className="mb-8">
              <div className="flex justify-between items-center p-4 rounded-lg shadow-md backdrop-blur-md bg-gradient-to-br from-[rgba(0,0,0,0.2)] to-[rgba(255,255,255,0.1)]">
                <h3 className="text-2xl font-bold text-white">
                  Molecules Generated from{" "}
                  {startingMolecule.name || startingMolecule.smile}
                </h3>
                <button
                  onClick={() => toggleSection(moleculeIndex)}
                  className="bg-blue-500 text-white px-4 py-2 rounded-lg"
                >
                  {expandedSections[moleculeIndex] ? "Minimize" : "Expand"}
                </button>
              </div>

              {expandedSections[moleculeIndex] && (
                <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                  {result.images.map((src, index) => (
                    <div
                      key={index}
                      className="backdrop-blur-md bg-gradient-to-br from-[rgba(0,0,0,0.2)] to-[rgba(255,255,255,0.1)] p-4 rounded-lg shadow-lg cursor-pointer max-w-full flex flex-col justify-center items-center"
                      onClick={() =>
                        handleClick(result.smiles[index])
                      }
                    >
                      <h4 className="text-lg font-semibold text-center mb-2 text-white">
                        Molecule {index + 1}
                      </h4>
                      <h4 className="text-lg font-semibold text-center mb-2 text-white break-words w-full">
                        {result.smiles[index]}
                      </h4>
                      <div className="flex items-center justify-center relative w-full">
                        <img
                          src={`data:image/png;base64,${src}`}
                          alt={`Molecule Image ${index + 1}`}
                          className="object-cover w-full aspect-square"
                        />
                      </div>
                    </div>
                  ))}
                  {result.images.length === 0 && <div className="text-center text-white w-full">No molecules generated, try a different molecule-effect combination.</div>}

                </div>
              )}

              {loadingMoleculeGens[moleculeIndex] && (
                <div className="flex justify-center items-center w-full h-64">
                  <div className="flex flex-col items-center">
                    <div className="w-12 h-12 border-4 border-white border-t-transparent rounded-full animate-spin"></div>
                    <p className="mt-4 text-white text-lg font-semibold">
                      Generating...
                    </p>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
