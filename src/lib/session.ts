import { getServerSession } from "next-auth";
import { authOptions } from "./auth";
import type { Session } from "next-auth";

/**
 * Safely get session - returns null if auth fails (e.g. DB/env not configured).
 * Use this instead of getServerSession directly to prevent "Something went wrong" on deploy.
 */
export async function getSafeSession(): Promise<Session | null> {
  try {
    return await getServerSession(authOptions);
  } catch {
    return null;
  }
}
