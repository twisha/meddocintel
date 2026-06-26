import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { auth } from "./api";

export function useAuthGuard() {
  const router = useRouter();
  useEffect(() => {
    if (!auth.isAuthenticated()) {
      router.replace("/login");
    }
  }, [router]);
  return auth.isAuthenticated();
}
