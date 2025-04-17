"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { FaUser, FaStar, FaDashcube } from "react-icons/fa"; // Import icons

export function TopBar() {
  const [isHovered, setIsHovered] = useState(false);

  const router = useRouter();

  const handleLoadDashboard = () => {
    router.push("/dashboard");
  };


  return (
    <div className="fixed top-4 left-0 right-0 flex items-center justify-end px-8 z-[9999]">

      <div className="flex items-center space-x-4">

        <div
          className={`flex items-center transition-all duration-300 z-[9999] rounded-full ${
            isHovered ? "px-2 py-2 shadow-md bg-gray-800 bg-opacity-90" : ""
          }`}
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
        >
          <button
            className={`w-8 h-8 flex items-center justify-center rounded-full focus:outline-none
              bg-green-600 text-white
            `}
            onClick={handleLoadDashboard}
          >
            <FaDashcube className="text-xl" size={16}/>
          </button>

            <div className="ml-2">
                <button
                  className="text-white text-sm focus:outline-none hover:underline"
                  onClick={handleLoadDashboard}
                >
                  Dashboard
                </button>
              
            </div>
        </div>
      </div>
    </div>
  );
}
