"use client";

import { useState } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Menu, Map, BookOpen, FileText, FolderKanban, Bot, Search } from "lucide-react";

const navLinks = [
  { href: "/search", label: "Search", icon: Search },
  { href: "/roadmaps", label: "Roadmaps", icon: Map },
  { href: "/courses", label: "Courses", icon: BookOpen },
  { href: "/guides", label: "Guides", icon: FileText },
  { href: "/projects", label: "Projects", icon: FolderKanban },
  { href: "/tutor", label: "AI Tutor", icon: Bot },
];

export function MobileNav() {
  const [open, setOpen] = useState(false);

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="ghost" size="icon" className="md:hidden">
          <Menu className="h-5 w-5" />
        </Button>
      </SheetTrigger>
      <SheetContent side="right" className="w-[280px] sm:w-72 max-w-[85vw]">
        <nav className="flex flex-col gap-4 mt-8">
          {navLinks.map((link) => {
            const Icon = link.icon;
            return (
              <Link
                key={link.href}
                href={link.href}
                onClick={() => setOpen(false)}
                className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium hover:bg-accent"
              >
                <Icon className="h-4 w-4" />
                {link.label}
              </Link>
            );
          })}
        </nav>
      </SheetContent>
    </Sheet>
  );
}
