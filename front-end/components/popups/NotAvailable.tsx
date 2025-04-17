
import React from 'react';

export default function NotAvailablePopup({setShowingUnavailable}:any) {
    return (
      <>
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-70 backdrop-blur-md px-4">
          <div className="relative bg-white p-10 rounded-3xl shadow-2xl max-w-sm w-full text-center transform transition-all duration-500 opacity-0 animate-in">

            <button
              type="button"
              onClick={() => {
                setShowingUnavailable(false);
              }}
              className="absolute top-4 right-4 text-gray-500 hover:text-gray-700 transition-colors"
              aria-label="Close overlay"
            >
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
  
            <img
              src="/assets/E.png"
              alt="Matched Stay Logo"
              className="w-20 h-20 mx-auto mb-6 rounded-full"
            />
  
            <h2 className="text-xl font-semibold text-gray-800 mb-2">
              This feature is not available
            </h2>
            <h3 className="text-lg text-black font-thin mb-8">
                We will be releasing this feature soon. Stay tuned!
            </h3>
          </div>
        </div>
  
        <style jsx>{`
          @keyframes fadeInZoom {
            0% {
              opacity: 0;
              transform: scale(0.95);
            }
            100% {
              opacity: 1;
              transform: scale(1);
            }
          }
          .animate-in {
            animation: fadeInZoom 0.5s ease-out forwards;
          }
        `}</style>
      </>
    );
  };