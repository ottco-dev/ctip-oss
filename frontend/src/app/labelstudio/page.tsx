"use client";
import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function LabelStudioRedirect() {
  const router = useRouter();
  useEffect(() => { router.replace("/annotation"); }, [router]);
  return null;
}
