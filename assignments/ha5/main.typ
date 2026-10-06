// ---------- Page & text setup (article, a4paper, 12pt, a4wide) ----------
#set page(paper: "a4", margin: (x: 2.5cm, y: 2.5cm), numbering: "1")
#set text(size: 12pt, lang: "en")
#set par(justify: true)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#set figure(placement: none)

// Colorful links (hyperref)
#show link: set text(fill: blue)
#show ref: set text(fill: blue)

// ---------- Custom commands ----------
// Indicator function: write #ind in math, e.g. $ #ind (x > 0) $
#let ind = $bb(1)$

// Author names / points: #authorpoints("Sadegh, W5 Monday", 8)
#let authorpoints(who, points) = text(size: 14.4pt)[
  #raw("(" + str(points) + " points)") #raw("[" + who + "]")
]

// ---------- Code style (lstlisting "mystyle") ----------
#show raw.where(block: true): it => block(
  width: 100%,
  fill: rgb("#F7F7F7"),
  stroke: 0.5pt + rgb("#EEEEEE"),
  inset: 8pt,
  radius: 0pt,
  text(size: 10pt, it),
)
#set raw(tab-size: 4, lang: "python")

// ---------- Title ----------
#align(center)[
  #v(1em)
  #text(size: 17pt, weight: "bold")[Machine Learning A (2026) \ Home Assignment 5]
  #v(1em)
  #text(size: 14pt, fill: red)[Ludovico Maria Spitaleri, DMH249]
  #v(1em)
]

// Please leave the table of contents as is, for the ease of navigation for TAs
#outline()
// #pagebreak() // Start a new page after the table of contents


= On the Role of Dependence #authorpoints("Sadegh, W5 Monday", 8)


= On Confidence Intervals #authorpoints("Sadegh, W5 Monday", 12)

== Part 1

== Part 2

= Loss Range Correction in Generalization Bounds #authorpoints("Sadegh, W5 Monday", 10)

== (i)

== (ii)


= Convolutional Neural Networks #authorpoints("Christian, W5 Friday", 70)

== Sobel filter #authorpoints("Christian, W5 Friday", 24)


== Convolutional neural networks #authorpoints("Christian, W5 Friday", 24)


== Augmentation #authorpoints("Christian, W5 Friday", 22)


// ---------- Placeholder figure with two subfigures ----------
// // Replace the rects with image("your-file.png", width: 100%)
// #let placeholder = rect(width: 100%, height: 4cm, fill: luma(230))

// #grid(
//   columns: (0.45fr, 0.45fr),
//   column-gutter: 1fr,
//   [
//     #figure(
//       placeholder,
//       caption: [Lorem Ipsum A],
//       kind: "subfigure",
//       supplement: none,
//       numbering: "(a)",
//     ) <fig:subfig_a>
//   ],
//   [
//     #figure(
//       placeholder,
//       caption: [Lorem Ipsum B],
//       kind: "subfigure",
//       supplement: none,
//       numbering: "(a)",
//     ) <fig:subfig_b>
//   ],
// )


// ---------- Placeholder figure ----------
// #figure(
//   rect(width: 50%, height: 4cm, fill: luma(230)),
//   // Dummy caption generated with lorem
//   caption: [#lorem(15)],
// ) <fig:placeholder>


// ---------- Placeholder code ----------
// ```python
// # Example code
// # Creating an example array
// data = np.array([5, 2, 8, 1, 6])

// # Calculating cumulative sum using cumsum
// cumulative_sum = np.cumsum(data)
// ```


// ---------- Bibliography ----------
// If you have references, put a bibliography.bib (or .yml) next to this file
// and uncomment:
// #bibliography("bibliography.bib")
