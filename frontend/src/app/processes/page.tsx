"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function ProcessesRedirect() {
  const router = useRouter();
  useEffect(() => {
    router.replace("/system?tab=processes");
  }, [router]);
  return null;
}
