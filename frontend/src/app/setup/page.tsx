"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function SetupRedirect() {
  const router = useRouter();
  useEffect(() => {
    router.replace("/system?tab=setup");
  }, [router]);
  return null;
}
