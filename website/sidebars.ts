import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'index',
    'getting-started',
    {
      type: 'category',
      label: 'SQL functions',
      items: [
        'functions/emeans',
        'functions/assign',
        'functions/stream',
        'functions/spatial',
        'functions/recipe-and-trace',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: ['reference/settings', 'reference/algorithm'],
    },
    'performance',
    'demos',
    'development',
  ],
};

export default sidebars;
