"use client";

import React, { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Loader2, FlaskConical, CheckCircle, FileText, ArrowRight, Save } from "lucide-react";

export default function SynthesisRoutePage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <SynthesisRouteContent />
    </Suspense>
  );
}

function SynthesisRouteContent() {
    const router = useRouter();
    const searchParams = useSearchParams();
    const [loadingText, setLoadingText] = useState("Initializing Synthesis...");
    const [progress, setProgress] = useState(0);
    const [completed, setCompleted] = useState(false);
    const [synthesisRoute, setSynthesisRoute] = useState(null);
    const [stepImages, setStepImages] = useState({});
    const [error, setError] = useState(null);

    const synthesisStages = [
        "Initializing Synthesis...",
        "Analyzing Molecular Structure...",
        "Predicting Reaction Pathways...",
        "Optimizing Chemical Reagents...",
        "Validating Synthesis Feasibility...",
        "Finalizing Synthesis Plan...",
        "Synthesis Complete!"
    ];

    useEffect(() => {
        const smiles = searchParams.get("smiles");
        if (!smiles) {
            setError("SMILES string is missing.");
            return;
        }

        fetchSynthesisRoute(smiles);

        let index = 0;
        const interval = setInterval(() => {
            if (index < synthesisStages.length) {
                setLoadingText(synthesisStages[index]);
                setProgress((prev) => Math.min(prev + 20, 100));
                index++;
            } else {
                clearInterval(interval);
                setCompleted(true);
            }
        }, 1500);

        return () => clearInterval(interval);
    }, [searchParams]);

    const fetchSynthesisRoute = async (smiles) => {
        try {
            const response = await fetch("https://erudite-backend-798229031686.europe-west2.run.app/synthesis", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ smiles }),
            });

            const data = await response.json();
            if (response.ok) {
                setSynthesisRoute(data);
                fetchStepImages(data.steps);
            } else {
                setError(data.error || "Failed to generate synthesis route.");
            }
        } catch (err) {
            setError("Error contacting the server.");
        }
    };

    const fetchStepImages = async (steps) => {
        try {
            const imageRequests = steps.flatMap((step, index) => [
                step.starting_material_smiles
                    ? { key: `step-${index + 1}-start`, smiles: step.starting_material_smiles }
                    : null,
                step.intermediate_smiles || step.final_product_smiles
                    ? { key: `step-${index + 1}-product`, smiles: step.intermediate_smiles || step.final_product_smiles }
                    : null,
            ]).filter(Boolean); 

            console.log("Image Requests:", imageRequests);

            const responses = await Promise.all(imageRequests.map((req) =>
                fetch("https://erudite-backend-798229031686.europe-west2.run.app/generateImage", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ smiles: req.smiles }),
                }).then((res) => res.json().then((data) => ({ key: req.key, image: data.image })))
            ));

            const imageMap = {};
            responses.forEach(({ key, image }) => {
                if (image) imageMap[key] = `data:image/png;base64,${image}`;
            });

            setStepImages(imageMap);
        } catch (err) {
            console.error("Error fetching images:", err);
        }
    };


    const content = () => {
        return (
          <div className="bg-gray-800 p-6 rounded-lg shadow-lg w-auto text-center">
      
            {error ? (
              <p className="text-red-400">{error}</p>
            ) : synthesisRoute ? (
              <>
                <div className="space-y-8 flex flex-col items-center">
                  {synthesisRoute.steps?.map((step, index) => (
                    <div
                      key={index}
                      className="flex flex-col md:flex-row items-center justify-center gap-4"
                    >
                      <div className="flex items-center gap-4">
                        <div className="p-4 bg-gray-900 rounded-lg flex flex-col items-center w-auto h-auto">
                          <p className="text-white font-semibold text-center text-sm">
                            {step.starting_material}
                          </p>
                          {stepImages[`step-${index + 1}-start`] ? (
                            <img
                              src={stepImages[`step-${index + 1}-start`]}
                              alt={step.starting_material}
                              className="w-24 h-24 mt-2"
                            />
                          ) : (
                            <div className="flex items-center justify-center w-24 h-24 bg-gray-700 text-white text-sm">
                              No Image
                            </div>
                          )}
                        </div>
      
                        <ArrowRight className="text-blue-400 w-8 h-8" />
      
                        <div className="p-4 bg-gray-900 rounded-lg flex flex-col items-center w-auto h-auto">
                          <p className="text-white font-semibold text-center text-sm">
                            {step.intermediate || step.final_product}
                          </p>
                          {stepImages[`step-${index + 1}-product`] ? (
                            <img
                              src={stepImages[`step-${index + 1}-product`]}
                              alt={step.intermediate || step.final_product}
                              className="w-32 h-auto mt-2 object-contain"                            />
                          ) : (
                            <div className="flex items-center justify-center w-24 h-24 bg-gray-700 text-white text-sm">
                              No Image
                            </div>
                          )}
                        </div>
                      </div>
                      <div className="text-white text-sm text-center">
                        <b>Reagents:</b> {step.reagents} <br /><br />
                        <b>Yield:</b> {step.yield}
                      </div>
                    </div>
                  ))}
                </div>
      
                {synthesisRoute.steps?.length > 0 && (
                  <div className="mt-8 p-6 bg-gray-900 rounded-lg flex flex-col items-center">
                    <h3 className="text-xl font-semibold text-white mb-2">
                      Final Product
                    </h3>
                    <p className="text-lg text-white font-bold">
                      {synthesisRoute.steps[synthesisRoute.steps.length - 1]
                        .final_product_smiles || "Unknown"}
                    </p>
                    {stepImages[`step-${synthesisRoute.steps.length}-product`] ? (
                      <img
                        src={stepImages[`step-${synthesisRoute.steps.length}-product`]}
                        alt="Final Product"
                        className="w-32 h-32 mt-4"
                      />
                    ) : (
                      <div className="flex items-center justify-center w-32 h-32 bg-gray-700 text-white text-sm mt-4">
                        No Image
                      </div>
                    )}
                  </div>
                )}
      
                <button
                  className="flex self-center mx-auto items-center justify-center gap-2 px-4 py-2 mt-6 bg-green-500 text-white rounded-lg hover:bg-green-600 transition"
                  onClick={() => {
                    router.push("/dashboard");
                  }}>
                  To dashboard
                  
                </button>
              </>
            ) : (
              <p className="text-red-400">Loading viable synthesis route</p>
            )}
          </div>
        );
      };
      

    const loadingSection = () => {
        return (
          <div>
            <p className="text-lg text-gray-300 mb-6">
              Simulating synthesis for the given molecule...
            </p>
      
            <div className="relative flex items-center justify-center w-full h-24 mb-6">
              {!completed ? (
                <FlaskConical className="w-20 h-20 animate-pulse text-white" />
              ) : (
                <CheckCircle className="w-20 h-20 text-green-400 animate-bounce" />
              )}
            </div>
      
            <div className="relative w-full h-6 bg-gray-700 rounded-full overflow-hidden mb-4">
              <div
                className="absolute left-0 top-0 h-full bg-gradient-to-r from-blue-500 to-green-400 transition-all duration-500"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
      
            <p className="text-lg font-semibold tracking-wide text-gray-100 animate-pulse">
              {loadingText}
            </p>
      
            <div className="flex justify-center mt-2">
              {!completed ? (
                <Loader2 className="w-10 h-10 animate-spin mt-6 text-gray-400" />
              ) : null}
            </div>
          </div>
        );
      };
      
      return (
        <div className="flex flex-col items-center justify-center min-h-screen text-white px-6">
          <div className="text-center">
            <h1 className="text-4xl font-bold mb-4 text-white">Synthesis Route</h1>
            {completed ? content() : loadingSection()}
          </div>
        </div>
      );
      
}
