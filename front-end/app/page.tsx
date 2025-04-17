"use client";
import React, { useState, useEffect } from "react";


export default function Home() {
  const [fadeIn, setFadeIn] = useState(false);

  useEffect(() => {
    setTimeout(() => setFadeIn(true), 300); 
  }, []);
  return (

      <div className="relative min-h-screen flex flex-col items-center justify-center text-white p-8 sm:p-20 font-[family-name:var(--font-geist-sans)]">
        <main className={`flex flex-col gap-8 rounded-xl p-8 w-full items-center relative z-10 transition-opacity duration-1000 ${fadeIn ? 'opacity-100' : 'opacity-0'}`}>
          <div className="text-center flex flex-col items-center">
            <h1 className="text-6xl font-bold text-center">Erudite</h1>
            <p className="text-xl text-gray-200 text-center">Empowering Innovation</p>
          </div>
          <div className="flex flex-row gap-4 w-full justify-center">
          <a
            className="shadow shadow-white text-white font-extrabold rounded-xl max-w-1/4 w-1/4 border border-solid border-transparent transition-colors flex items-center justify-center bg-black text-background gap-2 hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5"
            href="/setup"
          >
            Begin
          </a>
          <a
            className="shadow shadow-black text-black font-extrabold w-1/3 rounded-xl max-w-1/4 w-1/4 border border-solid border-transparent transition-colors flex items-center justify-center bg-white text-background gap-2 hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5"
            href="/dashboard"
          >
            Dashboard
          </a>
          </div>
        </main>
      </div>
  );
}
