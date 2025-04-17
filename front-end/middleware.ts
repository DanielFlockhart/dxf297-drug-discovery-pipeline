import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";



export function middleware(request: NextRequest) {
  return NextResponse.next();
}

// Match all routes except static files and API routes
export const config = {
  matcher: [
    "/((?!_next|api|favicon.ico|static).*)", // Exclude static paths
  ],
};
