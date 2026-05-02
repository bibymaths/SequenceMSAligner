window.MathJax = {
    tex: {
        // Defines inline math delimiters
        inlineMath: [["\\(", "\\)"], ["$", "$"]],

        // Defines block math delimiters (This covers your $$ and \[ requirements)
        displayMath: [["\\[", "\\]"], ["$$", "$$"]],

        processEscapes: true,
        processEnvironments: true
    },
    options: {
        ignoreHtmlClass: ".*|",
        processHtmlClass: "arithmatex"
    }
};