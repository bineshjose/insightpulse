import { redirect } from "next/navigation";

/** Root route: the product lives at /dashboard (login guard applies). */
export default function RootPage() {
  redirect("/dashboard");
}
