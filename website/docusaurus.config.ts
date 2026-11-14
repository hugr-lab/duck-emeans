import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// Docs for the hugr-lab DuckDB emeans extension, with the setup of the mssql
// extension's site (hugr-lab/mssql-extension, website/) so the two look alike.
// .github/workflows/pages.yml publishes the build at the root of
// https://hugr-lab.github.io/duck-emeans/ and the browser demos under /demo/;
// static/playground.html redirects the playground's old address there.
//
// VERSIONING: at each release run `npm run docusaurus docs:version <X.Y.Z>` in
// website/ and commit the snapshot. Docusaurus then serves the latest released
// version at the root and the live docs/ tree as "Next" under /next/. Until the
// first snapshot the live docs serve at the root.

const config: Config = {
  title: 'DuckDB emeans Extension',
  tagline: 'e-means clustering in SQL: mini-batch k-means with EMA updates and fitted scaling laws.',
  favicon: 'img/favicon.ico',

  url: 'https://hugr-lab.github.io',
  baseUrl: '/duck-emeans/',
  trailingSlash: true,

  organizationName: 'hugr-lab',
  projectName: 'duck-emeans',

  onBrokenLinks: 'throw',
  onBrokenAnchors: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: '/',
          editUrl: 'https://github.com/hugr-lab/duck-emeans/tree/main/website/',
          showLastUpdateTime: true,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themes: ['@docusaurus/theme-mermaid'],

  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },

  themeConfig: {
    metadata: [
      {name: 'keywords', content: 'DuckDB, e-means, k-means, clustering, embeddings, vector search, IVF, extension, SQL'},
      {name: 'description', content: 'DuckDB extension for e-means clustering: cluster vectors, embeddings and points in SQL, in memory or streamed from a query larger than memory.'},
    ],
    navbar: {
      title: 'emeans Extension',
      logo: {
        alt: 'Hugr Lab',
        src: 'img/logo-circle.svg',
        href: '/',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          href: 'https://hugr-lab.github.io/duck-emeans/demo/playground.html',
          label: 'Playground',
          position: 'left',
        },
        {
          href: 'https://hugr-lab.github.io/',
          label: 'Hugr Lab',
          position: 'right',
        },
        {
          href: 'https://github.com/hugr-lab/duck-emeans',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    colorMode: {
      defaultMode: 'light',
      disableSwitch: true,
      respectPrefersColorScheme: false,
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {label: 'Getting Started', to: '/getting-started/'},
            {label: 'SQL functions', to: '/functions/emeans/'},
            {label: 'Settings and limits', to: '/reference/settings/'},
          ],
        },
        {
          title: 'Try it',
          items: [
            {label: 'Playground', href: 'https://hugr-lab.github.io/duck-emeans/demo/playground.html'},
            {label: 'Demo', href: 'https://hugr-lab.github.io/duck-emeans/demo/'},
          ],
        },
        {
          title: 'Community',
          items: [
            {label: 'GitHub', href: 'https://github.com/hugr-lab/duck-emeans'},
            {label: 'Issues', href: 'https://github.com/hugr-lab/duck-emeans/issues'},
            {label: 'e-means paper', href: 'https://arxiv.org/abs/2611.09501'},
          ],
        },
        {
          title: 'Hugr Lab',
          items: [
            {label: 'Main site', href: 'https://hugr-lab.github.io/'},
            {label: 'Impressum', href: 'https://hugr-lab.github.io/impressum/'},
            {label: 'Datenschutz', href: 'https://hugr-lab.github.io/datenschutz/'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Hugr Lab.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['sql', 'bash'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
