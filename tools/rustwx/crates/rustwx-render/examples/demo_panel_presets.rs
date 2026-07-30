use rustwx_render::weather::{
    ECAPE_SEVERE_PANEL_PRODUCTS, SEVERE_CLASSIC_PANEL_PRODUCTS, WeatherProduct,
};

fn main() {
    print_panel("ecape-severe", &ECAPE_SEVERE_PANEL_PRODUCTS);
    print_panel("severe-classic", &SEVERE_CLASSIC_PANEL_PRODUCTS);
}

fn print_panel(name: &str, products: &[WeatherProduct]) {
    println!("{name}:");
    for product in products {
        println!("  {} -> {}", product.slug(), product.display_title());
    }
}
