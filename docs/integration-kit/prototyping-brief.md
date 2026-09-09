# Prototyping brief: explore a new ArWen experience

You are helping design a weather exploration and forecast creation application around ArWen 2.7. Use the attached integration guide, source/product/physics catalogs, and example client as the technical boundary. First read them and identify any gaps that prevent a design from being implemented. Do not invent a simulation API or claim a mocked workflow is connected.

The product gives equal care to exploring regular weather models and creating personal forecasts. It covers saved and running forecasts, historical periods, simulations, and future valid periods supported by the selected inputs. The current design direction is a restrained Frutiger Aero interface: clear glass-like surfaces, sky and water colors, airy depth, readable controls, and a large scientific weather map. Use the available space well. The main areas are Explore, Create forecast, and My forecasts, with contextual controls and a persistent time player.

Create three substantially different interface concepts. Show each as a useful working composition at desktop size, including a compact window. Explore alternative organization and interaction rather than changing only colors. Include high-quality visual treatments and artwork where they help orientation or make the product inviting, while keeping the map and its data dominant. Show how the visual system would be implemented in Rust egui; identify any effects that need custom rendering or assets.

For the strongest concept, build a clickable prototype with these journeys:

1. Choose a regular model and date, inspect several fields, select an exact valid time, and play a loop with a UTC range.
2. Create an own forecast with a clear source, area or tropical-cyclone target, duration, cadence, physics, and computer; review once and start.
3. Return to a saved or running forecast, switch domain/field/time, pause while output arrives, and reconnect after closing the client.
4. Request a sounding or 3D analysis from the selected frame, with an honest preparation state.

Use real catalog entries where provided. Keep forecast draft state separate from the pinned results viewer. Preserve exact timestamps, data-source attribution, projection, map scale, units, legends, contours, and wind barbs. Treat a map drawn small inside an oversized empty plotting frame as a defect. Keep the special HRRR maximum demonstration out; ordinary HRRR remains available when its source supports the requested period.

Use authentic screenshots and data only when supplied. Label mock data and simulated progress explicitly. Do not fabricate a completed forecast, operational performance numbers, storms, availability, or successful execution. For a prototype disconnected from the engine, show exactly which interactions are placeholders and which CLI/interface calls would make them real.

Stress-test the concept with awkward cases: a narrow window, a thousand subhourly frames, one unavailable field, a slow first-frame conversion, a moved inner domain, a disconnected computer, a target changed after review, an empty forecast list, and a source with incomplete historical coverage. Explain what the person sees and how they continue.

Return the prototype and assets, an annotated interaction map, a short implementation plan mapped to the documented interfaces, and a prioritized critique of weaknesses. Distinguish a visual preference from a usability defect, a missing engine capability, and a scientific correctness issue. Invite further exploration by presenting the most promising unresolved design questions with concrete alternatives.
