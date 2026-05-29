"use client";
import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function ExperimentsRedirect() {
  const router = useRouter();
  useEffect(() => { router.replace("/training?tab=experiments"); }, [router]);
  return null;
}
