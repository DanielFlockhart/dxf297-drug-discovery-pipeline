"use client";
import React from "react";
import { useRouter } from "next/navigation";
import { BiDna, BiAnalyse, BiData } from "react-icons/bi"; 
import { FaHeartbeat, FaPills, FaChartLine, FaBitcoin, FaStar } from "react-icons/fa";
import NotAvailablePopup from "@/components/popups/NotAvailable";

export default function Dashboard() {
  const router = useRouter();
  const [showingUnavailable, setShowingUnavailable] = React.useState(false);

  const sections = [
    {
      title: "Erudite Chemistry",
      isPremium: false,
      colorScheme: [
        "from-green-500 to-green-600",
        "from-green-400 to-green-500",
        "from-green-300 to-green-400",
      ],
      items: [
        {
          title: "Generation & Analysis",
          items: [
            { title: "Generate Molecules", path: "/setup", icon: <BiDna />,isAvailable: true  },
            { title: "Analyze", path: "/analytics", icon: <BiAnalyse />,isAvailable: true  },
          ],
        },
      ],
    },


    
  ];

  const dashboardItem = ({ title, path, icon, color,isAvailable }, index) => (
    <div
      key={index}
      className={`p-4 rounded-lg flex flex-row-reverse items-center justify-between cursor-pointer transform transition-all duration-300 
      bg-gradient-to-br ${color} shadow-lg hover:scale-105 hover:shadow-2xl hover:border hover:border-blue-500`}
      onClick={() => {
        if (isAvailable) {
          router.push(path);
        } else {
          setShowingUnavailable(true);
        }}}
    >
      <span className="text-2xl text-white">{icon}</span>
      <h1 className="text-xl font-semibold text-left text-white">{title}</h1>
    </div>
  );

  const subsection = ({ title, items }, index, colorScheme) => (
    <div key={index} className="mb-8">
      {/* Subsection Title */}
      <h3 className="text-2xl font-semibold mb-4">{title}</h3>
      {/* Subsection Items */}
      <div className="grid grid-cols-2 gap-6">
        {items.map((item, itemIndex) =>
          dashboardItem({ ...item, color: colorScheme[itemIndex % colorScheme.length] }, itemIndex)
        )}
      </div>
    </div>
  );

  const sectionOverlay = () => (
    <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center rounded-lg flex-col z-50">
      <span className="text-white text-xl font-bold">Upgrade to Premium to unlock</span>
      <button
  className="flex items-center mt-4 px-4 py-2 text-sm font-semibold text-white bg-gradient-to-r from-purple-500 to-blue-500 rounded-full hover:opacity-90 focus:outline-none"
  onClick={() => {
    router.push("/premium");
  }}
>
  <span>Upgrade to Premium</span>
  <FaStar className="text-lg ml-2" />
</button>

    </div>
  );

  return (
    <div className="relative flex flex-col text-white p-8 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <h1 className="text-7xl font-thin text-center mb-4">Erudite</h1>
      <h1 className="text-6xl font-bold text-center mb-8">Dashboard</h1>

      <div className="flex flex-col w-full max-w-5xl mx-auto">
        {sections.map((section, sectionIndex) => (
          <div
            key={sectionIndex}
            className="relative mb-12 p-6 border rounded-lg backdrop-blur-md bg-gradient-to-br from-[rgba(0,0,0,0.2)] to-[rgba(255,255,255,0.1)]"
          >
            {section.isPremium && sectionOverlay()}

            <h2
              className={`text-4xl font-bold mb-6 ${
                section.isPremium ? "opacity-50" : ""
              }`}
            >
              {section.title}
            </h2>

            <div
              className={`space-y-8 ${
                section.isPremium ? "opacity-50 pointer-events-none" : ""
              }`}
            >
              {section.items.length > 0 ? (
                section.items.map((subsectionItem, subsectionIndex) =>
                  subsection(subsectionItem, subsectionIndex, section.colorScheme)
                )
              ) : (
                <p className="text-xl italic text-gray-400">No items available.</p>
              )}
            </div>
          </div>
        ))}
      </div>
      
      {showingUnavailable && <NotAvailablePopup setShowingUnavailable={setShowingUnavailable} />}
    </div>
  );
}
