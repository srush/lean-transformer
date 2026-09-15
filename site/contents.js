// Reuse Verso's generated heading links so the contents follows source edits.
function placeContents() {
  const toc = document.querySelector('.page-toc');
  const introduction = document.querySelector('.code-content > .verso-text.mod-doc');
  if (!toc || !introduction) return;
  toc.querySelectorAll('li:not(.toc-level-1)').forEach(item => item.remove());
  const articleTitle = introduction.querySelector('h1');
  toc.querySelectorAll('li.toc-level-1').forEach(item => {
    const link = item.querySelector('a');
    if (articleTitle && link && decodeURIComponent(link.hash.slice(1)) === articleTitle.id) {
      item.remove();
    }
  });
  toc.querySelector('.page-toc-title').textContent = 'Table of contents';
  introduction.after(toc);
}
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', placeContents, { once: true });
} else {
  placeContents();
}
