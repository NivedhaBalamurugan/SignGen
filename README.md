American Sign Language Generation using Variational and Adversarial Learning

Project folder structure

```bash
├── repository/
│ ├── Dataset/
│ | ├── extended_words.json
│ │ ├── landmarks/
| │ │ ├── augmented/
| │ │ ├── merged/
| │ │ ├── palm_body/
| │ │ ├── final/
│ │ ├── videos/
| | ├── glove/
| | | └── glove.6B.50d.txt
│ │ └── chunks.json
│ │ └── missing.txt
| | └── wlasl_class_list.txt
| | └── WLASL_v0.3.json
│ ├── Models/

```

Landmark Structure

```json
{
    "<gloss_1>" : [
        <video1>[
            <frame1>[
                <landmark1>[x,y,z],
                <landmark2>[x,y,z],
                ...
                <landmark49>[x,y,z],
            ],
            <frame2>[],
            <frame3>[],
            ...
        ],
        <video2>[],
        <video3>[],
        ...
    ],
    "<gloss_2>" : [
        <video1>[],
        <video2>[],
        ...
    ],
    ...
}
```
