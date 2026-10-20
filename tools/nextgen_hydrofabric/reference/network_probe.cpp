// REFEREE TOOL ONLY. Calls pinned native ngen parser; never install/package.
// Apache-2.0. No native source modified or copied into the production reader.
#include "geopackage.hpp"
#include <fstream>
#include <iostream>
int main(int argc,char**argv){
    if(argc!=3){std::cerr<<"network_probe INPUT.gpkg OUTPUT.tsv\n";return 2;}
    try{
        std::ofstream out(argv[2]);if(!out)throw std::runtime_error("cannot open capture TSV");
        out<<"layer\tid\ttoid\n";
        for(const std::string layer:{"divides","flowpaths","nexus"}){
            auto features=ngen::geopackage::read(argv[1],layer,{});
            for(int i=0;i<features->get_size();++i){auto f=features->get_feature(i);
                out<<layer<<'\t'<<f->get_id()<<'\t'<<(f->has_property("toid")?f->get_property("toid").as_string():"")<<'\n';
            }
        }
        if(!out)throw std::runtime_error("native capture write failed");
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
