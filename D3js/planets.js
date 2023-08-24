const svg = d3.select("#svg-container")
    .append("svg")
    .attr("width", 1400)
    .attr("height", 2000);

// // Define the data
d3.json("planets-data.json").then(function (jsonData) {
    // Store the data in a variable
    const data = jsonData;

    const earthData = data.find(d => d.name === "Earth");

    const slider = d3.select("#slider")
        .on("input", function () {

            window.scrollTo({
                top: 1150,
                behavior: "smooth"
            });

            const sliderValue = this.value;

            if (sliderValue <= 2000) {
                equivalentValues(sliderValue);
                radiusScale.domain([d3.min(data, d => d.distance_light_year), d3.max(data, d => d.distance_light_year)])
                    .range([1, radiusOfUniverse]);
                bubbles.attr("r", d => radiusScale(d.distance_light_year) * radiusOfPlanet)
                    .attr("transform", (d, i) => {
                        const angle = i * angleStep;
                        const x = Math.cos(angle) * radiusScale(d.distance_light_year) * distanceOfPlanetFromEarth;
                        const y = Math.sin(angle) * radiusScale(d.distance_light_year) * distanceOfPlanetFromEarth;
                        return `translate(${x}, ${y})`;
                    }
                    );
                labels.attr("transform", (d, i) => {
                    const angle = i * angleStep;
                    const x = Math.cos(angle) * radiusScale(d.distance_light_year) * distanceOfPlanetFromEarth;
                    const y = Math.sin(angle) * radiusScale(d.distance_light_year) * distanceOfPlanetFromEarth;
                    return `translate(${x}, ${y})`;
                }
                );
            }

            else {
                radiusScale.domain([d3.min(data, d => d.distance_light_year), d3.max(data, d => d.distance_light_year)])
                    .range([1, 300000000]);
                bubbles.attr("r", d => radiusScale(d.distance_light_year) * 2)
                    .attr("transform", (d, i) => {
                        const angle = i * angleStep;
                        const x = Math.cos(angle) * radiusScale(d.distance_light_year) * 180;
                        const y = Math.sin(angle) * radiusScale(d.distance_light_year) * 180;
                        return `translate(${x}, ${y})`;
                    });
                labels.attr("transform", (d, i) => {
                    const angle = i * angleStep;
                    const x = Math.cos(angle) * radiusScale(d.distance_light_year) * 180;
                    const y = Math.sin(angle) * radiusScale(d.distance_light_year) * 180;
                    return `translate(${x}, ${y})`;
                });
            }
        });


    // Define the scales for each variable
    const userRadiusOfUniverse = d3.scaleLinear()
        .domain([100, 3000000000])
        .range([1, 3000000000]);

    const userRadiusOfPlanet = d3.scaleLinear()
        .domain([1, 5])
        .range([1, 30000]);

    const userDistanceOfPlanetFromEarth = d3.scaleLinear()
        .domain([5, 100])
        .range([1, 1000]);

    let radiusOfUniverse = 1;
    let radiusOfPlanet = 1;
    let distanceOfPlanetFromEarth = 1;

    // Define the color scale
    const colorScale = d3.scaleThreshold()
        .domain([4500, 6100, 10000])
        .range(['#6e5d01', '#FFD700', '#FFFFF0']);

    // Create the first legend group
    const legend1 = svg.append("g")
        .attr("class", "legend")
        .attr("transform", "translate(20, 20)");


    // Create the colored rectangles and labels for the first legend
    legend1.selectAll("rect")
        .data(colorScale.range())
        .enter()
        .append("rect")
        .attr("x", (d, i) => i * 80)
        .attr("y", 0)
        .attr("width", 50)
        .attr("height", 20)
        .attr("fill", d => d);

    legend1.selectAll("text")
        .data(colorScale.domain())
        .enter()
        .append("text")
        .attr("x", (d, i) => i * 80 + 20)
        .attr("y", 35)
        .attr("text-anchor", "middle")
        .attr("fill", "white")
        .text((d) => "< " + d);

    legend1.append("text")
        .attr("x", 0)
        .attr("y", -10)
        .attr("text-anchor", "start")
        .style("font-size", "14px")
        .attr("fill", "white")
        .style("font-weight", "bold")
        .text("Host Star Temperature (K)");

    // To get the equivalent values for all three variables for a given value between 1 and 3,000,000,000:
    const equivalentValues = (value) => {
        radiusOfUniverse = userRadiusOfUniverse.invert(value);
        radiusOfPlanet = userRadiusOfPlanet.invert(value);
        distanceOfPlanetFromEarth = userDistanceOfPlanetFromEarth.invert(value);
    };


    // Calculate the angle between each bubble
    const angleStep = 2 * Math.PI / data.length;

    // Define the scales
    const radiusScale = d3.scaleLinear()
        .domain([d3.min(data, d => d.distance_light_year), d3.max(data, d => d.distance_light_year)])
        .range([1, 100]); // min 100, max 3,000,000,000

    // Draw the bubbles
    const bubbles = svg.selectAll("circle")
        .data(data)
        .enter()
        .append("circle")
        .attr("cx", 700)
        .attr("cy", 800)
        .attr("r", d => radiusScale(d.distance_light_year) * 1) // min 1, max 5
        .attr("fill", d => colorScale(d.host_star_temperature))
        .attr("opacity", 0.8)
        .attr("stroke", "white")
        .attr("stroke-width", 1)
        .attr("transform", (d, i) => {
            const angle = i * angleStep;
            const x = Math.cos(angle) * radiusScale(d.distance_light_year) * 5; // min 5, max 100
            const y = Math.sin(angle) * radiusScale(d.distance_light_year) * 5; // distance from center
            return `translate(${x}, ${y})`;
        });

    // Add labels
    const labels = svg.selectAll("text")
        .data(data)
        .enter()
        .append("text")
        .text(d => `${d.name}`)
        .attr("x", 700)
        .attr("y", 780)
        .attr("font-size", 12)
        .attr("fill", "white")
        .attr("text-anchor", "middle")
        .attr("alignment-baseline", "central")
        .attr("transform", (d, i) => {
            const angle = i * angleStep;
            const x = Math.cos(angle) * radiusScale(d.distance_light_year) * 5;
            const y = Math.sin(angle) * radiusScale(d.distance_light_year) * 5;
            return `translate(${x}, ${y})`;
        })
        .style("visibility", "hidden");


    // Add event listeners to show/hide labels on click
    bubbles.on("click", (event, d) => {
        const tooltipName = document.getElementById("name");
        const tooltipDistance = document.getElementById("distance");
        const tooltipMass = document.getElementById("mass");
        const tooltipRadius = document.getElementById("radius");
        const tooltipPeriod = document.getElementById("period");
        const tooltipSemiMajorAxis = document.getElementById("semi_major_axis");
        const tooltipTemperature = document.getElementById("temperature");
        const tooltipHostStarTemperature = document.getElementById("host_star_temperature");
        const tooltipHostStarMass = document.getElementById("host_star_mass");

        tooltip.innerHTML = `<strong>Details about ${d.name}:</strong>`;
        tooltipName.innerHTML = `Name: ${d.name}`;
        tooltipDistance.innerHTML = `Distance: ${d.distance_light_year} ly from Earth`;
        tooltipMass.innerHTML = `Planet mass: ${d.mass} Jupiters`;
        tooltipRadius.innerHTML = `Planet radius: ${d.radius} Jupiters`;
        tooltipPeriod.innerHTML = `Planet period: ${d.period} Earth days`;
        tooltipSemiMajorAxis.innerHTML = `Semi-major axis: ${d.semi_major_axis} AU`;
        tooltipTemperature.innerHTML = `Planet temperature: ${d.temperature} K`;
        tooltipHostStarTemperature.innerHTML = `Host star temperature: ${d.host_star_temperature} K`;
        tooltipHostStarMass.innerHTML = `Host star mass: ${d.host_star_mass} Suns`;

        const tooltipEarth = document.getElementById("tooltip-earth");
        const tooltipNameEarth = document.getElementById("name-earth");
        const tooltipDistanceEarth = document.getElementById("distance-earth");
        const tooltipMassEarth = document.getElementById("mass-earth");
        const tooltipRadiusEarth = document.getElementById("radius-earth");
        const tooltipPeriodEarth = document.getElementById("period-earth");
        const tooltipSemiMajorAxisEarth = document.getElementById("semi_major_axis-earth");
        const tooltipTemperatureEarth = document.getElementById("temperature-earth");
        const tooltipHostStarTemperatureEarth = document.getElementById("host_star_temperature-earth");
        const tooltipHostStarMassEarth = document.getElementById("host_star_mass-earth");

        tooltipEarth.innerHTML = `<strong>Details about ${earthData.name}:</strong>`;
        tooltipNameEarth.innerHTML = `Name: ${earthData.name}`;
        tooltipDistanceEarth.innerHTML = `Distance: ${earthData.distance_light_year} ly from Earth`;
        tooltipMassEarth.innerHTML = `Planet mass: ${earthData.mass} Jupiters`;
        tooltipRadiusEarth.innerHTML = `Planet radius: ${earthData.radius} Jupiters`;
        tooltipPeriodEarth.innerHTML = `Planet period: ${earthData.period} Earth days`;
        tooltipSemiMajorAxisEarth.innerHTML = `Semi-major axis: ${earthData.semi_major_axis} AU`;
        tooltipTemperatureEarth.innerHTML = `Planet temperature: ${earthData.temperature} K`;
        tooltipHostStarTemperatureEarth.innerHTML = `Host star temperature: ${earthData.host_star_temperature} K`;
        tooltipHostStarMassEarth.innerHTML = `Host star mass: ${earthData.host_star_mass} Sun`;

        // Define your data as an array of objects with two properties: value1 and value2.
        const compare_data = [{ key: d.name, value1: d.mass, value2: d.temperature }, { key: earthData.name, value1: earthData.mass, value2: earthData.temperature }];
        // Set the dimensions and margins of the chart.

        const margin = { top: 20, right: 20, bottom: 60, left: 100 };
        const width = 500 - margin.left - margin.right;
        const height = 400 - margin.top - margin.bottom;

        const existingSvg = d3.select("#barchart svg");
        if (existingSvg) {
            existingSvg.remove();
        }

        const svg = d3.select("#barchart")
            .insert("svg", ":first-child")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom);

        // Append a group for the chart.
        const g = svg.append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        const colorScaleForPlanet = d3.scaleThreshold()
            .domain([260, 320, 600])
            .range(['blue', 'steelblue', 'red']);


        // Create the first legend group
        const legend2 = svg.append("g")
            .attr("class", "legend")
            .attr("transform", "translate(" + (width - 300) + "," + (height + margin.top + margin.bottom - 20) + ")");

        // Create the colored rectangles and labels for the first legend
        legend2.selectAll("rect")
            .data(colorScaleForPlanet.range())
            .enter()
            .append("rect")
            .attr("x", (d, i) => i * 60)
            .attr("y", 6)
            .attr("width", 20)
            .attr("height", 20)
            .attr("fill", d => d);

        legend2.selectAll("text")
            .data(colorScaleForPlanet.domain())
            .enter()
            .append("text")
            .attr("x", (d, i) => i * 60 + 10)
            .attr("y", 0)
            .attr("text-anchor", "middle")
            .attr("fill", "white")
            .style("font-size", "10px")
            .text((d) => "< " + d);

        legend2.append("text")
            .attr("x", 0)
            .attr("y", -10)
            .attr("text-anchor", "start")
            .style("font-size", "12px")
            .attr("fill", "white")
            .text("Planet Temperature (K)");

        // Add the title to the plot
        svg.append("text")
            .attr("x", (width / 2))
            .attr("y", margin.top / 2)
            .attr("text-anchor", "middle")
            .style("font-size", "12px")
            .attr("fill", "white")
            .attr("font-weight", "bold")
            .text(`Mass of ${d.name} compared to Earth`);

        // Set the scales for the x and y axes.
        const x = d3.scaleBand()
            .range([0, width])
            .padding(0.1)
            .domain(compare_data.map(d => d.key));

        const y = d3.scaleLinear()
            .range([height, 0])
            .domain([0, d3.max(compare_data, d => Math.max(d.value1))]);

        // Append the bars to the chart.
        g.selectAll(".bar")
            .data(compare_data)
            .enter().append("rect")
            .attr("class", "bar")
            .attr("fill", d => colorScaleForPlanet(d.value2))
            .attr("stroke", "white")
            .attr("x", d => x(d.key))
            .attr("y", d => y(d.value1))
            .attr("width", x.bandwidth())
            .attr("height", d => height - y(d.value1));

        // Append the x axis.
        g.append("g")
            .attr("transform", "translate(0," + height + ")")
            .call(d3.axisBottom(x));

        // Append the y axis.
        g.append("g")
            .call(d3.axisLeft(y));

        // Change the color of the axis lines
        g.selectAll("path")
            .attr("stroke", "white");

        window.scrollTo({
            top: 200,
            behavior: 'smooth'  // smooth scroll
        });


    });

    // Add event listeners to show/hide labels on mouseover/mouseout
    bubbles.on("mouseover", (event, d) => {
        labels.filter(data => data === d)
            .style("visibility", "visible");
    });
    bubbles.on("mouseout", (event, d) => {
        labels.filter(data => data === d)
            .style("visibility", "hidden");
    });


}).catch(function (error) {
    // Handle any errors loading the file
    console.error(error);
});