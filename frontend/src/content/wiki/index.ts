import type { WikiPage } from './types';

import home from './pages/home';
import installationLinux from './pages/installation-linux';
import installationWindows from './pages/installation-windows';
import installationMacos from './pages/installation-macos';
import setupWizard from './pages/setup-wizard';
import infrastructure from './pages/infrastructure';
import dataCollection from './pages/data-collection';
import labeling from './pages/labeling';
import training from './pages/training';
import inference from './pages/inference';
import fineTuning from './pages/fine-tuning';
import architecture from './pages/architecture';
import apiReference from './pages/api-reference';
import troubleshooting from './pages/troubleshooting';

/** All wiki pages in display order. */
export const WIKI_PAGES: WikiPage[] = [
  // setup section
  home,
  installationLinux,
  installationWindows,
  installationMacos,
  setupWizard,
  infrastructure,
  // workflow section
  dataCollection,
  labeling,
  training,
  inference,
  fineTuning,
  // reference section
  architecture,
  apiReference,
  troubleshooting,
];

export const WIKI_PAGE_MAP: Record<string, WikiPage> = Object.fromEntries(
  WIKI_PAGES.map((p) => [p.slug, p]),
);

export function getWikiPage(slug: string): WikiPage | undefined {
  return WIKI_PAGE_MAP[slug];
}

export function getWikiPagesBySection(section: WikiPage['section']): WikiPage[] {
  return WIKI_PAGES.filter((p) => p.section === section);
}
