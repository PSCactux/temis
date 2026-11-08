# Vendored front-end libraries

Served by the app itself at `/vendor/…`, so the web UI works without internet access.

- `tinymce/` — TinyMCE 6.8.6 (MIT, see `tinymce/license.txt`), from the `tinymce` npm package: core,
  the Silver theme, the DOM model, default icons, the Oxide skins, the content skins and the plugins
  the editor uses (lists, advlist, table, link, autolink, searchreplace, wordcount, charmap, visualblocks,
  pagebreak, quickbars, autoresize). `tinymce/langs/es.js` is the official Spanish language pack for
  TinyMCE 6, redistributed under the same MIT terms (via the `tinymce-i18n` npm package).
