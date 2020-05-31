
let canvas;

let noff = 0;

let coords = [];

let model;
let verticies;

function setup() {

    canvas = createCanvas(window.innerWidth, window.innerHeight);
    background(0);

    model = createModel();


}

function draw() {

    background(0);

    for(let coord of coords) {

        strokeWeight(10);
        stroke(255);
        point(map(coord.x, 0, 1, 0, width), map(coord.y, 0, 1, 0, height));

    }

    tf.tidy(() => {

        (async () => {
            frameRate(0);
            let xs = tf.tensor1d(coords.map(elm => elm.x));
            let ys = tf.tensor1d(coords.map(elm => elm.y));
            let fitHistory = await model.fit(xs, ys);
            xs.dispose();
            ys.dispose();
            frameRate(60);
        })();

        // let x1 = 0;
        // let x2 = 1;
        // let prediction = model.predict(tf.tensor1d([x1, x2]));
        // let data = prediction.arraySync();
        // let y1 = data[0][0];
        // let y2 = data[1][0];
        // strokeWeight(5);
        // line(map(x1, 0, 1, 0, width), map(y1, 0, 1, 0, height), map(x2, 0, 1, 0, width), map(y2, 0, 1, 0, height));

        beginShape();
        noFill();
        stroke(255);
        verticies = [];
        for(let i = 0; i <= 1.1; i += 0.02) {

            let prediction = model.predict(tf.tensor1d([i]));
            let data = prediction.arraySync();
            let y = data[0][0];
            strokeWeight(5);
            let vert = {x: map(i, 0, 1, 0, width), y: map(y, 0, 1, 0, height)};
            verticies.push(vert);
            vertex(vert.x, vert.y);

        }
        endShape();
        beginShape();
        noFill();
        for(let vert of verticies) {
            
            noff += 0.01;
            stroke(255, 0, 0, noise(noff) * 255);
            vertex(vert.x + (noise(noff) * 8 + 7), vert.y + (noise(noff) * 8 + 7));

        }
        endShape();
        beginShape();
        noFill();
        stroke(0, 170, 255, 200);
        for(let vert of verticies) {
            
            noff += 0.01;
            stroke(0, 170, 255, noise(noff) * 255);
            vertex(vert.x - (noise(noff) * 8 + 7), vert.y - (noise(noff) * 8 + 7));

        }
        endShape();

    });

}

function createModel() {

    const model = tf.sequential();
    const hiddenLayer = tf.layers.dense({
        units: 4,
        inputShape: [1],
        activation: 'sigmoid'
    });
    const outputLayer = tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    });

    model.add(hiddenLayer);
    model.add(outputLayer);

    model.compile({
        loss: tf.losses.meanSquaredError,
        optimizer: tf.train.adam(0.1)
    });

    return model;

}

function mousePressed() {

    coords.push({
        x: map(mouseX, 0, width, 0, 1),
        y: map(mouseY, 0, height, 0, 1)
    });

}
