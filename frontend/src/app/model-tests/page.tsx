"use client";
import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function ModelTestsRedirect() {
  const router = useRouter();
  useEffect(() => { router.replace("/inference?tab=pipeline"); }, [router]);
  return null;
}
