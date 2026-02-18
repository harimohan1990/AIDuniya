"use client";

import { memo } from "react";
import Link from "next/link";
import { useSession, signIn, signOut } from "next-auth/react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Search, GraduationCap, BookOpen, Map, FolderKanban, Bot, Newspaper } from "lucide-react";
import { MobileNav } from "./mobile-nav";

function HeaderInner() {
  const { data: session, status } = useSession();

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/80 backdrop-blur-xl supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto flex h-14 sm:h-16 items-center justify-between px-4 sm:px-6 lg:px-8 max-w-7xl 2xl:max-w-8xl">
        <Link href="/" className="flex items-center gap-1.5 sm:gap-2 font-bold text-lg sm:text-xl shrink-0">
          <GraduationCap className="h-7 w-7 text-primary" />
          AICourseMap
        </Link>

        <nav className="hidden md:flex items-center gap-4 lg:gap-6">
          <Link
            href="/search"
            className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1"
          >
            <Search className="h-4 w-4" />
            Search
          </Link>
          <Link
            href="/roadmaps"
            className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1"
          >
            <Map className="h-4 w-4" />
            Roadmaps
          </Link>
          <Link
            href="/courses"
            className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1"
          >
            <BookOpen className="h-4 w-4" />
            Courses
          </Link>
          <Link
            href="/guides"
            className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
          >
            Guides
          </Link>
          <Link
            href="/blog"
            className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1"
          >
            <Newspaper className="h-4 w-4" />
            Blog
          </Link>
          <Link
            href="/projects"
            className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1"
          >
            <FolderKanban className="h-4 w-4" />
            Projects
          </Link>
          <Link
            href="/tutor"
            className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1"
          >
            <Bot className="h-4 w-4" />
            AI Tutor
          </Link>
        </nav>

        <div className="flex items-center gap-1 sm:gap-2 shrink-0">
          <MobileNav />
          <Link href="/search">
            <Button variant="ghost" size="icon" className="md:hidden">
              <Search className="h-4 w-4" />
            </Button>
          </Link>
          {status === "loading" ? (
            <div className="h-9 w-9 rounded-full bg-muted animate-pulse" />
          ) : session ? (
            <>
              <Link href="/dashboard" className="hidden sm:block">
                <Button variant="ghost" size="sm">
                  Dashboard
                </Button>
              </Link>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" className="relative h-9 w-9 rounded-full">
                    <Avatar className="h-9 w-9">
                      <AvatarImage src={session.user?.image ?? undefined} />
                      <AvatarFallback>
                        {session.user?.name?.[0] ?? session.user?.email?.[0] ?? "U"}
                      </AvatarFallback>
                    </Avatar>
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-56">
                  <DropdownMenuItem asChild>
                    <Link href="/dashboard">Dashboard</Link>
                  </DropdownMenuItem>
                  <DropdownMenuItem asChild>
                    <Link href="/dashboard/profile">Profile</Link>
                  </DropdownMenuItem>
                  <DropdownMenuItem asChild>
                    <Link href="/dashboard/settings">Settings</Link>
                  </DropdownMenuItem>
                  {((session.user as { role?: string })?.role) === "ADMIN" && (
                    <DropdownMenuItem asChild>
                      <Link href="/admin">Admin</Link>
                    </DropdownMenuItem>
                  )}
                  <DropdownMenuItem
                    onClick={() => signOut()}
                    className="text-destructive focus:text-destructive"
                  >
                    Sign out
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </>
          ) : (
            <>
              <Button variant="ghost" size="sm" asChild>
                <Link href="/auth/signin">Sign in</Link>
              </Button>
              <Button size="sm" asChild>
                <Link href="/auth/signup">Sign up</Link>
              </Button>
            </>
          )}
        </div>
      </div>
    </header>
  );
}

export const Header = memo(HeaderInner);
