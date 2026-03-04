# La' Partzuf

A browser-based face similarity analyzer.

**[Click here to start](https://davidunga.github.io/la-partzuf/)**

## Usage

Upload photos of three people (more photos → better accuracy):
- **Two references** - Define the endpoints of a similarity spectrum
- **One subject** - Gets positioned between the references based on similarity

## Technicals

The app detects faces and computes embeddings using face-api.js. Each person's embeddings are averaged, and cosine distances determine the subject's position on the spectrum.
All processing runs client-side.

## License

MIT
