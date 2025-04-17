"use client";
export const dynamic = "force-dynamic";
import React, { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
export default function OverviewPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <OverviewContent />
    </Suspense>
  );
}

function OverviewContent() {
  const searchParams = useSearchParams();
  const smiles = searchParams?.get("smiles");

  const [imageUrl, setImageUrl] = useState(null);
  const [molecule, setMolecule] = useState({
    smiles: smiles,
    classes: [],
    effects: [],
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const router = useRouter();

  useEffect(() => {
    if (!smiles) {
      setError("Missing SMILES data.");
      setLoading(false);
      return;
    }

    fetchMoleculeData();
  }, [smiles]);

  const fetchMoleculeData = async () => {
    try {
      setLoading(true);
      setError(null);

      const imageResponse = await fetch(
        "https://erudite-backend-798229031686.europe-west2.run.app/generateImage",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ smiles }),
        }
      );
      const imageData = await imageResponse.json();

      if (imageResponse.status !== 200) {
        throw new Error(imageData.error || "Failed to generate molecule image.");
      }
      setImageUrl(imageData.image);

      const classResponse = await fetch(
        "https://erudite-backend-798229031686.europe-west2.run.app/predictClasses",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ smiles }),
        }
      );
      const classData = await classResponse.json();

      const effectResponse = await fetch(
        "https://erudite-backend-798229031686.europe-west2.run.app/predictEffects",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ smiles }),
        }
      );
      const effectData = await effectResponse.json();

      if (classResponse.status !== 200 || effectResponse.status !== 200) {
        throw new Error(
          classData.error || effectData.error || "Failed to fetch predictions."
        );
      }

      // Update state with fetched data
      setMolecule({
        smiles: smiles,
        classes: Object.entries(classData.predicted_classes)
          .filter(([_, probability]) => probability > 0.02).
           sort((a, b) => b[1] - a[1]),
        effects: Object.entries(effectData.predicted_effects)
          .filter(([_, probability]) => probability > 0.02).

          sort((a, b) => b[1] - a[1]),
      });
    } catch (err) {
      setError(err.message || "An error occurred while processing the molecule.");
    } finally {
      setLoading(false);
    }
  };

  const getBackgroundColor = (probability) => {
    if (probability >= 80) return "bg-blue-600";
    if (probability >= 60) return "bg-blue-400";
    if (probability >= 40) return "bg-blue-400";
    if (probability >= 20) return "bg-blue-300";
    if (probability > 0) return "bg-blue-300";
    return "bg-gray-300"; 
  };

  const [newNote, setNewNote] = useState("");
  const [notes, setNotes] = useState([]);

  const handleAddNote = () => {
    if (!newNote) return;
    setNotes((prevNotes) => [...prevNotes, newNote]);
    setNewNote("");
  };

  

  return (
    <div>
      <div className="items-center justify-center min-h-screen p-8 text-white">

        <h1 className="text-3xl font-semibold mb-6 text-center">Molecule Overview</h1>

        {loading ? (
          <div className="flex justify-center items-center w-full h-64">
            <div className="flex flex-col items-center">
              <div className="w-12 h-12 border-4 border-white border-t-transparent rounded-full animate-spin"></div>
              <p className="mt-4 text-white text-lg font-semibold">Processing...</p>
            </div>
          </div>
        ) : error ? (
          <div className="text-center text-red-500">{error}</div>
        ) : (
          <>
            <div className="flex justify-center mb-6">
              {imageUrl && (
                <img
                  src={`data:image/png;base64,${imageUrl}`}
                  alt={`Image of ${molecule?.smiles}`}
                  width={400}
                  height={400}
                  className="rounded-lg shadow-lg"
                />
              )}
            </div>
            <div className="flex items-center justify-center">
              <div className="flex justify-center max-w-4xl w-full mt-4 mb-4 space-x-4">
                <button
                  className="w-2/5 rounded-lg transition-all duration-300 transform hover:scale-105 flex items-center justify-center bg-gradient-to-r from-blue-500 to-green-500 hover:from-blue-600 hover:to-green-600 text-white gap-2 h-12 px-8 shadow-lg hover:shadow-xl"
                  onClick={() => {
                    const string = `/molecule-synthesis?smiles=${encodeURIComponent(smiles)}`;
                    console.log(string);
                    router.push(string);
                  }
                    }>
                  Generate Synthesis Route
                  <span className="material-icons text-2xl">route</span>
                </button>
                
              </div>
            </div>
            <div className="max-w-4xl mx-auto  backdrop-blur-md  bg-gradient-to-br from-[rgba(0,0,0,0.2)] to-[rgba(255,255,255,0.1)] rounded-lg shadow-lg p-6 space-y-6">
              <h1 className="text-xs font-semibold mb-4 text-center">Unfortunately, this model has bias towards psychedelic tryptamine compounds. When doing statistical analysis on generated molecules and excluding the effects/class, the model is correctly predicteding at least a small confidence of molecules having the desired effects.</h1>
              <div>
                <h2 className="text-2xl font-semibold mb-2">SMILES</h2>
                <p className="text-lg">{molecule?.smiles}</p>
              </div>

              <div>
                <h2 className="text-2xl font-semibold mb-2">Predicted Molecular Classes</h2>
                <div className="grid grid-cols-2 gap-4">
                  {molecule.classes.map(([className, probability], index) => (
                    <div
                      key={index}
                      className={`${getBackgroundColor(
                        probability
                      )} rounded-lg p-4 flex justify-between items-center hover:bg-gray-400 transition-colors`}
                    >
                      <span className="text-lg">{className}</span>
                      <span className="text-lg font-semibold">
                        {probability}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h2 className="text-2xl font-semibold mb-2">Predicted Effects</h2>
                <div className="grid grid-cols-2 gap-4">
                  {molecule.effects.map(([effect, probability], index) => (
                    <div
                      key={index}
                      className={`${getBackgroundColor(
                        probability
                      )} rounded-lg p-4 flex justify-between items-center hover:bg-gray-400 transition-colors`}
                    >
                      <span className="text-lg">{effect}</span>
                      <span className="text-lg font-semibold">
                        {probability}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>


            </div>
            

          </>
        )}

        
      </div>
    </div>
  );
}
