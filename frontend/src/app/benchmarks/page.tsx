import { redirect } from 'next/navigation';

export default function BenchmarksPage() {
  redirect('/evaluation?tab=benchmarks');
}
