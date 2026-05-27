'use client';

import { createContext, useContext } from 'react';

export type Lang = 'en' | 'de' | 'es';

export interface WikiLangCtx {
  lang: Lang;
  setLang: (l: Lang) => void;
}

export const WikiLangContext = createContext<WikiLangCtx>({
  lang: 'en',
  setLang: () => {},
});

export function useWikiLang() {
  return useContext(WikiLangContext);
}
