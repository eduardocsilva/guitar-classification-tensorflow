import React, { useEffect, useState } from "react";
import logo from "./logo.svg";
import "./App.css";
import * as tf from "@tensorflow/tfjs";
import { makeStyles } from '@material-ui/core/styles';
import { Button, Card, CardActionArea, CardActions, CardContent, CardMedia, Typography } from "@material-ui/core"

const useStyles = makeStyles({
    root: {
        width: 400,
        height: 435,
        maxWidth: "90vw",
        marginBottom: 30,
        padding: 10
    },
    media: {
        height: 350
    }
});

function App() {
    const classes = useStyles();
    const modelUrl = "../../ml_model/model.json";
    const guitarModels = [
        "Fender Jaguar",
        "Fender Jazzmaster",
        "Fender Mustang",
        "Fender Stratocaster",
        "Fender Telecaster",
        "Gibson ES",
        "Gibson Flying V",
        "Gibson SG",
        "Gibson Explorer",
        "Gibson Firebird",
        "Gibson Les Paul"
    ]

    const loadModel = async (url: string) => {
        try {
            setModel(await tf.loadGraphModel(url));
        } catch (e) {
            console.log("An error occured:", e)
        }
    }

    const getClassification = async () => {
        const preprocesedImg = preprocess(document.querySelector("#selectedImage") as HTMLImageElement)
        const predictions = await model.predict(preprocesedImg).data();
        const normalizedPredictions = await tf.softmax(predictions).data();

        let sortedPredictions = Array.from(normalizedPredictions)
            .map(function (p: any, i: any) {
                return {
                    probability: p,
                    className: guitarModels[i]
                };
            }).sort(function (a: any, b: any) {
                return b.probability - a.probability;
            });

        const predictedClass = sortedPredictions[0];
        setClassification(predictedClass.className)
        setProbability(Math.round(predictedClass.probability * 10000) / 100)
    }

    const preprocess = (img: HTMLImageElement) => {
        return tf.browser.fromPixels(img)
            .resizeNearestNeighbor([180, 180])
            .toFloat()
            .expandDims();
    }

    const changeHandler = (event: any) => {
        setImage(URL.createObjectURL(event.target.files[0]));
        setImageSelected(true);
    };

    // Hooks
    const [model, setModel] = useState<any>();
    const [classification, setClassification] = useState<any>();
    const [probability, setProbability] = useState<any>();
    const [image, setImage] = useState<any>();
    const [imageSelected, setImageSelected] = useState<any>(false);

    useEffect(() => {
        tf.ready().then(() => {
            loadModel(modelUrl);
        });
    }, []);

    return (
        <div className="App">
            <header className="App-header">
                <div id="title">
                    <h3>Guitar Classification</h3>
                    <h6>(powered by Tensorflow.js)</h6>
                </div>
                {
                    imageSelected ? (
                        <Card id="prediction" className={classes.root}>
                            <CardMedia
                                className={classes.media}
                                image={image}
                                title="Guitar"
                            />
                            <img src={image} alt="" id="selectedImage" onLoad={getClassification} hidden />
                            {
                                (model && imageSelected) ? (
                                    <CardContent>
                                        <Typography gutterBottom variant="h5" component="h2">
                                            {classification}
                                        </Typography>
                                        <Typography variant="body2" color="textSecondary" component="p">
                                            With {probability}% of confidence
                                        </Typography>
                                    </CardContent>
                                ) : (
                                    <CardContent>
                                        <Typography variant="body2" color="textSecondary" component="p">
                                            Waiting for the classification...
                                        </Typography>
                                    </CardContent>
                                )
                            }
                        </Card>
                    ) : null
                }
                <Button
                    id="upload-image-btn"
                    variant="contained"
                    component="label"
                    color="secondary"
                >
                    {!imageSelected ? "Upload an image to classify" : "Upload another image"}
                    <input
                        type="file"
                        onChange={changeHandler}
                        hidden
                    />
                </Button>
            </header>
        </div>
    );
}

export default App;
